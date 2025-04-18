import copy
import os
import random
from itertools import product
from math import prod
from typing import List, Iterator, Tuple
import time
from typing import List, Dict, Tuple, Optional, Any, Iterator
import math
import torch
import torch.fx.experimental
import torch.fx.experimental.proxy_tensor
import torch.fx.experimental.symbolic_shapes

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
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


from collections import Counter
from typing import List, Iterator


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
    ModelCls: Any,
    model_args: Any,
    job_config: JobConfig,
    device: torch.device,
    train_spec: TrainSpec,
    world_size: int,
    parallelism_config: Optional[Dict[str, int]] = None,
) -> float:
    """
    Trace the model and measure forward pass latency.

    Args:
        job_config: Configuration for the job
        device: Device to run the model on
        parallelism_config: Optional custom parallelism configuration

    Returns:
        Forward pass latency in seconds
    """
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

    world_mesh = parallel_dims.build_mesh(device_type="cpu")

    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    with torch._subclasses.FakeTensorMode(
        static_shapes=True, allow_non_fake_inputs=True, shape_env=shape_env
    ):
        with device:
            model = ModelCls.from_model_args(model_args)

        assert not parallel_dims.pp_enabled
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        model = train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)

        logger.info(f"Model {model} created with {train_spec.name} spec")
        seq_length = model_args.max_seq_len

        tokens = torch.randint(
            0, 128256, (1, seq_length), dtype=torch.int32, device=device
        )

        # Measure forward pass latency
        start_time = time.time()
        make_fx_out = torch.fx.experimental.proxy_tensor.make_fx(
            model,
            tracing_mode="fake",
            decomposition_table={},
            _allow_non_fake_inputs=True,
            record_module_stack=True,
        )(tokens)
        end_time = time.time()

        latency = end_time - start_time
        print(f"Trace latency: {latency:.4f} seconds")

        return latency


def main() -> None:
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

    device = torch.device("cpu")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.distributed.init_process_group(
        backend="fake",
        world_size=world_size,
        store=FakeStore(),
        rank=int(os.environ.get("RANK", 0)),
    )

    train_spec = get_train_spec(job_config.model.name)
    ModelCls = train_spec.cls
    model_args = train_spec.config[job_config.model.flavor]
    model_args.update_from_config(job_config, tokenizer=MockTokenizer(128256))

    print(f"World size: {world_size}")
    print(f"Prime factorization: {prime_factorize(world_size)}")

    results = []

    # Try all possible parallelism configurations
    for i, config in enumerate(generate_parallelism_configs(world_size)):
        print(f"Testing configuration {i + 1}: {config}")
        latency = trace_model(
            ModelCls,
            model_args,
            job_config,
            device,
            train_spec,
            parallelism_config=config,
            world_size=world_size,
        )
        print(f"Configuration {config} latency: {latency:.4f} seconds")
        results.append((f"Config {i + 1}: {config}", latency))

    # Print summary of results
    print("\n===== RESULTS SUMMARY =====")
    results.sort(key=lambda x: x[1])  # Sort by latency
    for config_name, latency in results:
        print(f"{config_name}: {latency:.4f} seconds")


if __name__ == "__main__":
    main()
