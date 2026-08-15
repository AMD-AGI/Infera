###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import re
from typing import List, Optional

from infera.projection.core.projection.base_module_profiler import BaseModuleProfiler
from infera.projection.core.projection.profiler_spec import ModuleProfilerSpec
from infera.projection.core.projection.training_config import TrainingConfig

from .embedding import EmbeddingProfiler
from .layer_norm import LayerNormProfiler
from .output_layer import OutputLayerProfiler
from .transformer_layer import (
    get_dense_transformer_layer_profiler_spec,
    get_moe_transformer_layer_profiler_spec,
)


def build_profiler(spec: ModuleProfilerSpec, depth=0) -> BaseModuleProfiler:
    """
    Recursively build a profiler instance from a ModuleProfilerSpec.
    """
    if not issubclass(spec.profiler, BaseModuleProfiler):
        raise TypeError(f"spec.profiler must be subclass of BaseModuleProfiler, got {spec.profiler}")

    if depth == 0:
        print(f"Begin build profiler: {spec.profiler.__name__}")

    print(f"{'--'*(depth+1)}[{spec.profiler.__name__}]")

    sub_profilers = {}
    if spec.sub_profiler_specs:
        depth += 1
        for name, sub_spec in spec.sub_profiler_specs.items():
            if sub_spec is None:
                sub_profilers[name] = None
            elif isinstance(sub_spec, ModuleProfilerSpec):
                # build sub profiler with spec
                sub_profilers[name] = build_profiler(sub_spec, depth)
            elif issubclass(sub_spec, BaseModuleProfiler):
                # init sub profile
                print(f"{'--'*(depth+1)}[{sub_spec.__name__}]({name})")
                sub_profilers[name] = sub_spec(spec.config, sub_profilers=None)
            else:
                raise TypeError(f"Invalid type for sub_profiler_specs['{name}']: {type(sub_spec)}")

    return spec.profiler(config=spec.config, sub_profilers=sub_profilers)


def get_language_model_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=LanguageModelProfiler,
        config=config,
        sub_profiler_specs={
            "embedding": EmbeddingProfiler,
            "dense_transformer_layer": get_dense_transformer_layer_profiler_spec(config),
            "moe_transformer_layer": get_moe_transformer_layer_profiler_spec(config),
            "final_layernorm": LayerNormProfiler,
            "output_layer": OutputLayerProfiler,
        },
    )


def _get_balanced_layer_distribution(n_layers: int, total_stages: int) -> List[int]:
    """
    Distribute layers across stages as evenly as possible.
    Remainder layers are distributed to the first stages.

    Example: 61 layers, 4 stages -> [16, 15, 15, 15]
    """
    base_layers = n_layers // total_stages
    remainder = n_layers % total_stages

    layers_per_stage = []
    for i in range(total_stages):
        # First 'remainder' stages get one extra layer
        if i < remainder:
            layers_per_stage.append(base_layers + 1)
        else:
            layers_per_stage.append(base_layers)

    return layers_per_stage


def _get_explicit_layer_distribution(
    n_layers: int,
    total_stages: int,
    decoder_first: Optional[int],
    decoder_last: Optional[int],
) -> List[int]:
    """
    Get layer distribution with explicit first/last stage layer counts.
    Middle stages get evenly distributed remainder.
    """
    if total_stages == 1:
        return [n_layers]

    layers_per_stage = [0] * total_stages

    # Handle first and last stages
    first_layers = decoder_first if decoder_first is not None else 0
    last_layers = decoder_last if decoder_last is not None else 0

    # If not specified, they'll be computed from the middle distribution
    remaining_layers = n_layers - first_layers - last_layers
    middle_stages = (
        total_stages - 2
        if (decoder_first is not None and decoder_last is not None)
        else (total_stages - 1 if (decoder_first is not None or decoder_last is not None) else total_stages)
    )

    if middle_stages > 0 and remaining_layers > 0:
        base_middle = remaining_layers // middle_stages
        middle_remainder = remaining_layers % middle_stages

        # Fill middle stages
        start_idx = 1 if decoder_first is not None else 0
        end_idx = total_stages - 1 if decoder_last is not None else total_stages

        for i in range(start_idx, end_idx):
            local_idx = i - start_idx
            if local_idx < middle_remainder:
                layers_per_stage[i] = base_middle + 1
            else:
                layers_per_stage[i] = base_middle

    # Set first and last
    if decoder_first is not None:
        layers_per_stage[0] = first_layers
    elif remaining_layers > 0 and decoder_last is not None:
        # First was not specified but last was - first gets from distribution above
        pass

    if decoder_last is not None:
        layers_per_stage[-1] = last_layers

    return layers_per_stage


def _parse_layout_stage_layer_counts(
    layout: Optional[str], total_stages: int, n_layers: int
) -> Optional[List[int]]:
    """
    Parse Megatron-style pipeline layout into decoder-layer counts per virtual stage.

    Example layout:
      'Et*4|t*4|...|t*3,L'
    """
    if not layout:
        return None

    normalized = str(layout).strip()
    # Handle extra shell quoting, e.g. "'Et*4|...|t*3,L'"
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in ("'", '"'):
        normalized = normalized[1:-1].strip()

    # Split virtual stages by '|'; strip trailing non-decoder markers like ",L".
    stage_specs = [part.strip() for part in normalized.split("|") if part.strip()]
    if len(stage_specs) != total_stages:
        raise ValueError(
            f"pipeline_model_parallel_layout has {len(stage_specs)} stages, "
            f"but PP*VPP expects {total_stages} stages."
        )

    layers_per_stage: List[int] = []
    for spec in stage_specs:
        spec = spec.split(",", 1)[0].strip()
        matches = re.findall(r"[tT](?:\*(\d+))?", spec)
        if not matches:
            raise ValueError(f"Invalid pipeline stage spec '{spec}' in pipeline_model_parallel_layout.")
        layer_count = sum(int(m) if m else 1 for m in matches)
        layers_per_stage.append(layer_count)

    if sum(layers_per_stage) != n_layers:
        raise ValueError(
            "pipeline_model_parallel_layout decoder layer count mismatch: "
            f"layout has {sum(layers_per_stage)} decoder layers, model has {n_layers}."
        )

    return layers_per_stage


# language profiler spec -> build_profiler() -> language profiler -> run profiling methods
class LanguageModelProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        rank = int(os.getenv("RANK", "0"))
        self.layers = self.get_layers_for_rank(
            global_rank=rank,
            n_layers=self.config.model_config.num_layers,
            pp_size=self.config.model_parallel_config.pipeline_model_parallel_size,
            tp_size=self.config.model_parallel_config.tensor_model_parallel_size,
            cp_size=self.config.model_parallel_config.context_model_parallel_size,
            ep_size=self.config.model_parallel_config.expert_model_parallel_size,
            num_virtual_pipeline_stages=self.config.model_parallel_config.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_layout=self.config.model_parallel_config.pipeline_model_parallel_layout,
        )
        self._gemm_backend = None
        self._sdpa_backend = None

    def set_simulation_backends(self, gemm_backend=None, sdpa_backend=None):
        """Set simulation backends and propagate to all sub-profilers."""
        self._gemm_backend = gemm_backend
        self._sdpa_backend = sdpa_backend

        # Propagate to transformer layer sub-profilers (which further propagate
        # to attention, MLP, router sub-profilers).
        for key in ("dense_transformer_layer", "moe_transformer_layer"):
            if key in self.sub_profilers and self.sub_profilers[key] is not None:
                layer_profiler = self.sub_profilers[key]
                if hasattr(layer_profiler, "set_simulation_backends"):
                    layer_profiler.set_simulation_backends(gemm_backend, sdpa_backend)

        # Propagate to embedding (uses simple analytical estimate in sim mode).
        if "embedding" in self.sub_profilers and self.sub_profilers["embedding"] is not None:
            emb = self.sub_profilers["embedding"]
            if hasattr(emb, "set_simulation_mode"):
                emb.set_simulation_mode(gemm_backend is not None or sdpa_backend is not None)

        # Propagate GEMM backend to output layer (vocab projection GEMM).
        if "output_layer" in self.sub_profilers and self.sub_profilers["output_layer"] is not None:
            out = self.sub_profilers["output_layer"]
            if gemm_backend is not None and hasattr(out, "set_gemm_backend"):
                out.set_gemm_backend(gemm_backend)

    def set_inference_phase(self, phase, kv_seq_len=None):
        """Propagate a forward-only inference phase to layer profilers.

        See :meth:`AttentionProfiler.set_inference_phase`.  ``phase=None``
        restores default training behaviour.
        """
        for key in ("dense_transformer_layer", "moe_transformer_layer"):
            layer_profiler = self.sub_profilers.get(key)
            if layer_profiler is not None and hasattr(layer_profiler, "set_inference_phase"):
                layer_profiler.set_inference_phase(phase, kv_seq_len)

    def get_layers_for_rank(
        self,
        global_rank: int,
        n_layers: int,
        pp_size: int,
        tp_size: int,
        cp_size: int,
        ep_size: int,
        num_virtual_pipeline_stages: Optional[int] = None,
        decoder_first_pipeline_num_layers: Optional[int] = None,
        decoder_last_pipeline_num_layers: Optional[int] = None,
        pipeline_model_parallel_layout: Optional[str] = None,
    ) -> List[int]:
        """
        Get layers assigned to a specific rank, handling imbalanced layer distribution.

        When layers aren't evenly divisible by PP*VPP, distribute remainder layers
        to the first virtual stages (or use decoder_first/last_pipeline_num_layers if set).
        """
        vpp_size = num_virtual_pipeline_stages if num_virtual_pipeline_stages is not None else 1

        chunks = LanguageModelProfiler.get_virtual_stage_layers_for_rank(
            self,
            global_rank=global_rank,
            n_layers=n_layers,
            pp_size=pp_size,
            tp_size=tp_size,
            cp_size=cp_size,
            ep_size=ep_size,
            num_virtual_pipeline_stages=vpp_size,
            decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
            decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
            pipeline_model_parallel_layout=pipeline_model_parallel_layout,
        )
        return [layer for chunk in chunks for layer in chunk]

    @staticmethod
    def get_virtual_stage_layers_for_rank(
        self,
        global_rank: int,
        n_layers: int,
        pp_size: int,
        tp_size: int,
        cp_size: int,
        ep_size: int,
        num_virtual_pipeline_stages: Optional[int] = None,
        decoder_first_pipeline_num_layers: Optional[int] = None,
        decoder_last_pipeline_num_layers: Optional[int] = None,
        pipeline_model_parallel_layout: Optional[str] = None,
    ) -> List[List[int]]:
        """
        Get per-virtual-stage decoder layers assigned to a rank.
        """
        vpp_size = num_virtual_pipeline_stages if num_virtual_pipeline_stages is not None else 1
        total_stages = pp_size * vpp_size

        model_parallel_size = pp_size * tp_size * cp_size * ep_size
        model_parallel_rank = global_rank % model_parallel_size
        pp_rank = model_parallel_rank // (tp_size * cp_size * ep_size)

        # Check for explicit first/last pipeline layer counts
        # Try to get from self.config if available, otherwise use passed arguments
        decoder_first = decoder_first_pipeline_num_layers
        decoder_last = decoder_last_pipeline_num_layers
        if self is not None and hasattr(self, "config") and self.config is not None:
            mp_config = self.config.model_parallel_config
            if decoder_first is None:
                decoder_first = getattr(mp_config, "decoder_first_pipeline_num_layers", None)
            if decoder_last is None:
                decoder_last = getattr(mp_config, "decoder_last_pipeline_num_layers", None)
            if pipeline_model_parallel_layout is None:
                pipeline_model_parallel_layout = getattr(mp_config, "pipeline_model_parallel_layout", None)

        # Build layer counts per virtual stage
        if pipeline_model_parallel_layout:
            layers_per_stage = _parse_layout_stage_layer_counts(
                pipeline_model_parallel_layout, total_stages, n_layers
            )
        elif decoder_first is not None or decoder_last is not None:
            # Use explicit layer distribution
            layers_per_stage = _get_explicit_layer_distribution(
                n_layers, total_stages, decoder_first, decoder_last
            )
        else:
            # Auto-distribute: spread remainder layers across first stages
            layers_per_stage = _get_balanced_layer_distribution(n_layers, total_stages)

        # A physical pp_rank hosts multiple virtual stages in an interleaved fashion.
        # pp_rank 0 gets virtual stages: 0, pp_size, 2*pp_size, ...
        # pp_rank 1 gets virtual stages: 1, pp_size+1, 2*pp_size+1, ...
        my_virtual_stages = range(pp_rank, total_stages, pp_size)

        assigned_chunks: List[List[int]] = []
        for vs_index in my_virtual_stages:
            # Calculate start layer by summing layers in all previous stages
            start_layer = sum(layers_per_stage[:vs_index])
            count = layers_per_stage[vs_index]
            assigned_chunks.append(list(range(start_layer, start_layer + count)))

        return assigned_chunks

    def _estimate_layer_communication(self, layer_idx: int, layer_type: str):
        """
        Estimate communication overhead for a single layer.

        Args:
            layer_idx: Index of the layer
            layer_type: Type of layer ('dense' or 'moe')

        Returns:
            List of communication operations with time and message size
        """
        from infera.projection.core.projection.module_profilers import collective_model as cm
        from infera.projection.core.projection.module_profilers.collective_args import (
            get_default_args,
        )

        mp_config = self.config.model_parallel_config
        model_config = self.config.model_config
        runtime_config = self.config.runtime_config

        tp = mp_config.tensor_model_parallel_size
        pp = mp_config.pipeline_model_parallel_size
        ep = getattr(mp_config, "expert_model_parallel_size", 1)
        cp = getattr(mp_config, "context_model_parallel_size", 1)

        # Only estimate communication for EP (TP AllReduce is already in the benchmarked run)
        # PP communication is handled separately in pipeline simulation
        if ep == 1:
            return []

        # Get configuration
        hidden_size = model_config.hidden_size
        batch_size = runtime_config.micro_batch_size
        seq_len = runtime_config.sequence_length
        moe_router_topk = model_config.moe_router_topk

        # Setup collective model
        num_nodes = int(os.getenv("NNODES", "1"))
        gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))

        coll_args = get_default_args(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            tp=tp,
            pp=pp,
            ep=ep,
            cp=cp,
            hardware_config=None,
        )

        comm_ops = []

        # MoE All-to-All (if EP > 1 and this is a MoE layer)
        if ep > 1 and layer_type == "moe":
            tokens_per_batch = seq_len * batch_size
            dispatch_size = tokens_per_batch * hidden_size * moe_router_topk * 2  # BF16

            a2a_dispatch = cm.alltoall(coll_args, dispatch_size, ep, groups=["ep"])
            # dispatch time is same as combine time
            a2a_combine = a2a_dispatch

            # Forward: dispatch + combine, Backward: same
            fwd_time = (a2a_dispatch + a2a_combine) / 1000  # Convert to ms
            bwd_time = fwd_time  # Same as forward

            comm_ops.append(
                {
                    "type": "MoE All-to-All",
                    "time_fwd_ms": fwd_time,
                    "time_bwd_ms": bwd_time,
                    "message_size_mb": dispatch_size / (1024 * 1024),
                    "group_size": ep,
                }
            )

        return comm_ops

    def get_dp_size(self) -> int:
        num_nodes = int(os.getenv("NNODES", "1"))
        gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
        mp = self.config.model_parallel_config
        parallel_gpus = (
            mp.tensor_model_parallel_size
            * mp.context_model_parallel_size
            * mp.pipeline_model_parallel_size
            * mp.expert_model_parallel_size
        )
        if num_nodes == 1:
            # Minimum nodes to host the model-parallel mesh (ceil division).
            # Plain ``parallel_gpus // gpus_per_node`` is 0 when mesh fits on
            # one node (e.g. TP=PP=EP=CP=1), which breaks activation scaling.
            num_nodes = max(1, (parallel_gpus + gpus_per_node - 1) // gpus_per_node)
        world_size = num_nodes * gpus_per_node
        dp_size = world_size // mp.expert_model_parallel_size // mp.pipeline_model_parallel_size
        return max(1, dp_size)

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        total_params = 0
        if rank is None:
            layers = range(self.config.model_config.num_layers)
        else:
            layers = self.layers
        for layer in layers:
            is_moe = self.config.model_config.moe_pattern[layer]
            if is_moe:
                total_params += self.sub_profilers["moe_transformer_layer"].estimated_num_params(rank)
            else:
                total_params += self.sub_profilers["dense_transformer_layer"].estimated_num_params(rank)
        if 0 in self.layers:
            total_params += self.sub_profilers["embedding"].estimated_num_params(rank)
        if self.config.model_config.num_layers - 1 in self.layers:
            total_params += self.sub_profilers["final_layernorm"].estimated_num_params(rank)
            total_params += self.sub_profilers["output_layer"].estimated_num_params(rank)
        return total_params

