from collections import Counter
import copy
from operator import getitem
import os
from math import prod
from typing import List, Iterator, Tuple
import time
from typing import Dict, Optional
import torch
import torch.fx.experimental
import torch.fx.experimental.proxy_tensor
import torch.fx.experimental.symbolic_shapes
import asyncio

from torchtitan.components.ft import FTManager
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.llama3.model import Transformer, TransformerModelArgs
from torchtitan.protocols.train_spec import get_train_spec, TrainSpec
from torchtitan.fake.fake_pg import FakeStore
from torchtitan import fx_serialize
import aiofiles

# Import protoc-generated classes
from . import torch_titan_pb2 as pb


class MockTokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self._n_words = vocab_size

    def encode(self, *args, **kwargs) -> List[int]:
        return [0] * self._n_words

    def decode(self, *args, **kwargs) -> str:
        return " ".join(["word"] * self._n_words)


def prime_factorize(n: int) -> List[int]:
    """
    Compute the prime factorization of a number.

    Args:
        n: The number to factorize

    Returns:
        List of prime factors
    """
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n and n > 1:
            factors.append(n)
            break
    return factors


def lists_from_primes_gen(nums: List[int], d: int) -> Iterator[List[int]]:
    """
    Yield, without duplicates, every length‑d list whose ordered
    elements are products of disjoint subsets of `nums`.
    Empty subsets are allowed (slot = 1).
    """
    if d <= 0:
        raise ValueError("d must be positive")

    primes = list(Counter(nums).items())  # (prime, multiplicity)
    out = [1] * d  # current partial result

    def dfs(i: int):
        if i == len(primes):  # all primes placed
            yield out.copy()
            return

        p, c = primes[i]  # distribute c copies of p
        comp = [0] * d  # composition of c into d parts

        def comp_rec(pos: int, remaining: int):
            if pos == d - 1:  # last slot gets the rest
                comp[pos] = remaining
                for k in range(d):  # apply
                    if comp[k]:
                        out[k] *= p ** comp[k]
                yield from dfs(i + 1)
                for k in range(d):  # back‑track
                    if comp[k]:
                        out[k] //= p ** comp[k]
                return

            for cnt in range(remaining + 1):
                comp[pos] = cnt
                yield from comp_rec(pos + 1, remaining - cnt)

        yield from comp_rec(0, c)

    yield from dfs(0)


def generate_parallelism_configs(world_size: int) -> Iterator[Dict[str, int]]:
    fixed = {
        "dp_replicate": 1,
        "pp": 1,
    }

    search = ["dp_shard", "tp", "cp"]
    primes = prime_factorize(world_size)
    for assigsn in lists_from_primes_gen(primes, len(search)):
        config = {**fixed, **dict(zip(search, assigsn))}
        # Ensure that the product of all degrees equals world_size
        assert prod(config.values()) == world_size
        yield config

def parallel_config_to_str(config: Dict[str, int]) -> str:
    ws = prod(config.values())
    return f"ws{ws}cp{config['cp']}dp{config['dp_shard']}tp{config['tp']}pp{config['pp']}"

async def trace_model(
    *,
    model_cls: type[Transformer],
    model_args: TransformerModelArgs,
    batch_size: int,
    job_config: JobConfig,
    train_spec: TrainSpec,
    device: torch.device,
    world_size: int,
    parallelism_config: Optional[Dict[str, int]] = None,
) -> Tuple[pb.TraceResult, str, Dict[str, float]]:
    """
    Trace the model and measure forward pass latency.

    Args:
        model: Model instance to trace
        job_config: Configuration for the job
        parallelism_config: Optional custom parallelism configuration

    Returns:
        Tuple of (TraceResult protobuf, readable graph string, timing breakdown dict)
    """
    timing = {}
    start_total = time.time()

    # Use provided parallelism config or get from job_config
    if parallelism_config is None:
        pc = job_config.parallelism
    else:
        # Make a copy to avoid modifying the original
        pc = copy.deepcopy(job_config.parallelism)
        pc.data_parallel_shard_degree = parallelism_config["dp_shard"]
        pc.data_parallel_replicate_degree = parallelism_config["dp_replicate"]
        pc.context_parallel_degree = parallelism_config["cp"]
        pc.tensor_parallel_degree = parallelism_config["tp"]
        pc.pipeline_parallel_degree = parallelism_config["pp"]

    # Create a device mesh on meta device
    parallel_dims = ParallelDims(
        dp_shard=pc.data_parallel_shard_degree,
        dp_replicate=pc.data_parallel_replicate_degree,
        cp=pc.context_parallel_degree,
        tp=pc.tensor_parallel_degree,
        pp=pc.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not pc.disable_loss_parallel,
    )
    assert not parallel_dims.pp_enabled

    start_mesh = time.time()
    world_mesh = parallel_dims.build_mesh(device_type=device.type)
    timing["build_mesh"] = time.time() - start_mesh


    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    fake_mode = torch._subclasses.FakeTensorMode(
        static_shapes=True, allow_non_fake_inputs=True, shape_env=shape_env
    )
    with fake_mode, device:
        train_context = dist_utils.get_train_context(
            parallel_dims.loss_parallel_enabled,
            job_config.parallelism.enable_compiled_autograd,
        )
        loss_fn = train_spec.build_loss_fn(job_config)

        start_model = time.time()
        model = model_cls.from_model_args(model_args)
        timing["model_init"] = time.time() - start_model

        # Apply parallelism to the model
        start_parallelize = time.time()
        model = train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model_parts = [model]
        timing["parallelize"] = time.time() - start_parallelize

        seq_length = model.model_args.max_seq_len
        optimizers = train_spec.build_optimizers_fn(
            model_parts, job_config, FTManager(None)
        )
        # Create the model just once
        inputs = torch.randint(0, 128256, (batch_size // (parallel_dims.dp_shard * parallel_dims.dp_replicate), seq_length), dtype=torch.int32)
        labels = inputs.clone().detach().to(torch.long)
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in optimizers.model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in optimizers.model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        def step(opt_sd):
            optimizers.load_state_dict(opt_sd)
            optimizers.zero_grad()

            with train_context(optional_context_parallel_ctx):
                assert len(optimizers.model_parts) == 1
                pred = optimizers.model_parts[0](inputs)
                loss = loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()

            dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=world_mesh["pp"] if parallel_dims.pp_enabled else None,
            )
            optimizers.step()
            return optimizers.state_dict()

        start_tracing = time.time()
        make_fx_out = torch.fx.experimental.proxy_tensor.make_fx(
            step,
            tracing_mode="fake",
            decomposition_table={},
            _allow_non_fake_inputs=True,
            record_module_stack=True,
        )(optimizers.state_dict())
        # Remove unecessary nodes
        for node in make_fx_out.graph.nodes:
            if node.op == "placeholder" and "val" not in node.meta:
                make_fx_out.graph.erase_node(node)
        make_fx_out.graph.eliminate_dead_code()
        make_fx_out.recompile()
        timing["tracing"] = time.time() - start_tracing

        # Retrieve collective metadata
        for node in make_fx_out.graph.nodes:
            if node.op != "call_function":
                continue
            from torch.distributed._tools.fake_collectives import collective_ops, CollectiveOp
            import torch.distributed as dist
            if node.target == torch.ops._c10d_functional.wait_tensor.default:
                continue 
            if node.target in collective_ops:
                args = list(node.args)
                for i, a in enumerate(args):
                    if "_torchbind_obj" in getattr(a, "name", ""):
                        args[i] = getattr(make_fx_out, a.name)
                    elif getattr(a, "target", None) == getitem:
                        t = args[i].args[0]
                        args[i] = t.meta["val"][args[i].args[1]]
                    elif isinstance((val := getattr(a, "meta", {}).get("val")), torch._subclasses.FakeTensor):
                        args[i] = val
                group = CollectiveOp.get_process_group(node.target, args)
                res = node.meta["val"]
                size = CollectiveOp.get_comm_tensor_size(node.target, res, args, node.kwargs)
                node.meta["collective_meta"] = pb.CollectiveMeta(
                    group_ranks=dist.get_process_group_ranks(group),
                    comm_tensor_size=size,
                    group_desc=group.group_desc,
                    group_name=group.name(),
                )

        start_export = time.time()
        # fx_serialize.serialize now returns a protoc GraphModuleData
        graph_module_proto = fx_serialize.serialize(make_fx_out)
        timing["export"] = time.time() - start_export

        timing["total"] = time.time() - start_total

        # Create ParallelConfig protobuf message from ParallelDims
        parallel_config_proto = pb.ParallelConfig(
            dp_replicate=parallel_dims.dp_replicate,
            dp_shard=parallel_dims.dp_shard,
            cp=parallel_dims.cp,
            tp=parallel_dims.tp,
            pp=parallel_dims.pp,
            world_size=parallel_dims.world_size,
            enable_loss_parallel=parallel_dims.loss_parallel_enabled,
        )

        # Create TraceResult protobuf message
        result_proto = pb.TraceResult(
            parallel_dims=parallel_config_proto,
            graph_module=graph_module_proto
        )

        return result_proto, make_fx_out.print_readable(), timing


async def main() -> None:
    """Main function to run the simulation."""
    job_config = JobConfig()
    job_config.maybe_add_custom_args()
    job_config.parse_args()
    job_config.training.compile = False
    # TODO: can we relax some of these?
    job_config.parallelism.enable_async_tensor_parallel = False
    job_config.float8.enable_fsdp_float8_all_gather = False
    job_config.float8.precompute_float8_dynamic_scale_for_fsdp = False
    job_config.float8.recipe_name = "rowwise"

    device = torch.device("cuda")
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    setup_start = time.time()
    torch.distributed.init_process_group(
        backend="fake",
        world_size=world_size,
        store=FakeStore(),
        rank=int(os.environ.get("RANK", 0)),
    )
    setup_time = time.time() - setup_start
    print(f"Setup time: {setup_time:.4f} seconds")

    train_spec = get_train_spec(job_config.model.name)
    ModelCls = train_spec.cls
    model_args = train_spec.config[job_config.model.flavor]
    model_args.n_layers = 1
    model_args.update_from_config(job_config, tokenizer=MockTokenizer(128256))

    print(f"World size: {world_size}")
    print(f"Prime factorization: {prime_factorize(world_size)}")

    results = []
    batch_size = 2**22 // model_args.max_seq_len # 4 million token batch size
    assert batch_size > 0, "Batch size must be greater than 0"

    # Generate all configurations in advance to show progress
    configs = list(generate_parallelism_configs(world_size))
    configs = [c for c in configs if (c["tp"] <= model_args.n_kv_heads) and (c["dp_shard"] <= batch_size) and (c["cp"] <= 32)]
    total_configs = len(configs)

    # Try all possible parallelism configurations
    async def run_config(i: int, config: Dict[str, int]):
        print(f"Testing configuration {i + 1}/{total_configs}: {config}")
        # trace_model now returns a protoc TraceResult
        proto, code, timing = await trace_model(
            model_cls=ModelCls,
            model_args=model_args,
            batch_size=batch_size,
            job_config=job_config,
            train_spec=train_spec,
            world_size=world_size,
            parallelism_config=config,
            device=device,
        )
        # Serialize the protobuf message to bytes
        out_bytes = proto.SerializeToString()
        name = parallel_config_to_str(config)
        # Write binary protobuf data
        async with aiofiles.open(f"traces/{name}.pb", "wb") as f:
            await f.write(out_bytes)
        # Write readable graph code
        async with aiofiles.open(f"traces/{name}.py", "w") as f:
            await f.write(code)
        return f"Config {i}", timing

    os.makedirs("traces", exist_ok=True)
    # Run all configurations in parallel
    tasks = [run_config(i, config) for i, config in enumerate(configs)]
    #results = await asyncio.gather(*tasks)
    for task in tasks:
        results.append(await task)  

    # Print summary of results
    print("\n===== RESULTS SUMMARY =====")
    results.sort(key=lambda x: x[1]["total"])  # Sort by latency
    for config_name, timing in results:
        print(
            f"{config_name}: {''.join([f'{k}: {v:.4f} ' for k, v in timing.items()])}"
        )


if __name__ == "__main__":
    asyncio.run(main())
