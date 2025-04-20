from collections import Counter
import contextlib
import copy
import os
import random
from itertools import product
from math import prod
from typing import List, Iterator, Tuple
import time
from typing import List, Dict, Tuple, Optional, Any, Iterator
import torch
import torch.fx.experimental
import torch.fx.experimental.proxy_tensor
import torch.fx.experimental.symbolic_shapes

from torchtitan.components.ft import FTManager
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.llama3.model import Transformer, TransformerModelArgs
from torchtitan.protocols.train_spec import get_train_spec, TrainSpec
from torchtitan.tools.logging import logger
from torchtitan.fake.fake_pg import FakeStore


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

    search = ["cp", "dp_shard", "tp"]
    primes = prime_factorize(world_size)
    for assigsn in lists_from_primes_gen(primes, len(search)):
        config = {**fixed, **dict(zip(search, assigsn))}
        # Ensure that the product of all degrees equals world_size
        assert prod(config.values()) == world_size
        yield config


def trace_model(
    *,
    model_cls: type[Transformer],
    model_args: TransformerModelArgs,
    job_config: JobConfig,
    train_spec: TrainSpec,
    device: torch.device,
    world_size: int,
    parallelism_config: Optional[Dict[str, int]] = None,
) -> float:
    """
    Trace the model and measure forward pass latency.

    Args:
        model: Model instance to trace
        job_config: Configuration for the job
        parallelism_config: Optional custom parallelism configuration

    Returns:
        Tuple of (forward pass latency in seconds, timing breakdown dict)
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

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.parallelism.enable_compiled_autograd,
    )

    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    fake_mode = torch._subclasses.FakeTensorMode(
        static_shapes=True, allow_non_fake_inputs=True, shape_env=shape_env
    )
    with fake_mode, device:
        loss_fn = train_spec.build_loss_fn(job_config)


        start_model = time.time()
        model = model_cls.from_model_args(model_args)
        timing["model_init"] = time.time() - start_model

        # Apply parallelism to the model
        start_parallelize = time.time()
        model = train_spec.parallelize_fn(
            model, world_mesh, parallel_dims, job_config
        )
        model_parts = [model]
        timing["parallelize"] = time.time() - start_parallelize

        seq_length = model.model_args.max_seq_len
        optimizers = train_spec.build_optimizers_fn(
            model_parts, job_config, FTManager(None)
        )
        # Create the model just once
        inputs = torch.randint(
            0, 128256, (1, seq_length), dtype=torch.int32
        )
        labels = inputs.clone().detach().to(torch.long)
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        def step():
            optimizers.zero_grad()

            with train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                pred = model_parts[0](inputs)
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

        start_tracing = time.time()
        make_fx_out = torch.fx.experimental.proxy_tensor.make_fx(
            step,
            tracing_mode="fake",
            decomposition_table={},
            _allow_non_fake_inputs=True,
            record_module_stack=True,
        )()
        timing["tracing"] = time.time() - start_tracing

        timing["total"] = time.time() - start_total

        return timing


def main() -> None:
    """Main function to run the simulation."""
    overall_start = time.time()

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

    # Generate all configurations in advance to show progress
    configs = list(generate_parallelism_configs(world_size))
    total_configs = len(configs)

    # Try all possible parallelism configurations
    for i, config in enumerate(configs):
        print(f"Testing configuration {i + 1}/{total_configs}: {config}")
        timing = trace_model(
            model_cls=ModelCls,
            model_args=model_args,
            job_config=job_config,
            train_spec=train_spec,
            world_size=world_size,
            parallelism_config=config,
            device=device,
        )
        results.append((f"Config {i + 1}: {config}", timing))

    # Print summary of results
    print("\n===== RESULTS SUMMARY =====")
    results.sort(key=lambda x: x[1]["total"])  # Sort by latency
    for config_name, timing in results:
        print(f"{config_name}: {''.join([f'{k}: {v:.4f} ' for k, v in timing.items()])}")

if __name__ == "__main__":
    main()
