import copy
import os
import random
import torch
import torch.fx.experimental
import torch.fx.experimental.proxy_tensor
import torch.fx.experimental.symbolic_shapes

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.logging import logger
from torchtitan.fake.fake_pg import FakeStore

class MockTokenizer(Tokenizer):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self._n_words = vocab_size

    def encode(self, *args, **kwargs) -> list[int]:
        return [0] * self._n_words

    def decode(self, *args, **kwargs) -> str:
        return " ".join(["word"] * self._n_words)

def trace_model(job_config: JobConfig, device: torch.device):
    parallelism_config = job_config.parallelism

    # 2. Create a device mesh on meta device
    # Use a simple 1D mesh for TP simulation if PP=1, otherwise use 2D
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not parallelism_config.disable_loss_parallel,
    )

    torch.distributed.init_process_group(
        backend="fake",
        world_size=world_size,
        store=FakeStore(),
        rank=int(os.environ.get("RANK", 0)),
    )
    
    world_mesh = parallel_dims.build_mesh(device_type="cpu")
    train_spec = get_train_spec(job_config.model.name)
    ModelCls = train_spec.cls
    model_args = train_spec.config[job_config.model.flavor]
    model_args.update_from_config(job_config, tokenizer=MockTokenizer(128256))

    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    with torch._subclasses.FakeTensorMode(static_shapes=True, allow_non_fake_inputs=True, shape_env=shape_env):
        with device:
            model = ModelCls.from_model_args(model_args)

        loss_fn = train_spec.build_loss_fn(job_config)

        if parallel_dims.pp_enabled:
            if not train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                pp_schedule,
                model_parts,
                pp_has_first_stage,
                pp_has_last_stage,
            ) = train_spec.pipelining_fn(
                model,
                world_mesh,
                parallel_dims,
                job_config,
                device,
                model_args,
                train_spec.parallelize_fn,
                loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            # TODO: pp
            #del model

            #for m in self.model_parts:
            #    m.to_empty(device=init_device)
            #    with torch.no_grad():
            #        m.init_weights(buffer_device=buffer_device)
            #    m.train()

            ## confirm that user will be able to view loss metrics on the console
            #ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = train_spec.parallelize_fn(
                model, world_mesh, parallel_dims, job_config
            )
            pass

        logger.info(f"Model {model} created with {train_spec.name} spec")
        seq_length = model_args.max_seq_len

        tokens = torch.randint(
            0, 128256, (1, seq_length), dtype=torch.int32, device=device
        )
        make_fx_out = torch.fx.experimental.proxy_tensor.make_fx(model, tracing_mode="fake", decomposition_table={}, _allow_non_fake_inputs=True, record_module_stack=True)(tokens)
        print(make_fx_out.graph)


if __name__ == "__main__":
    job_config = JobConfig()
    job_config.maybe_add_custom_args()
    job_config.parse_args()
    trace_model(job_config, device=torch.device("cpu"))