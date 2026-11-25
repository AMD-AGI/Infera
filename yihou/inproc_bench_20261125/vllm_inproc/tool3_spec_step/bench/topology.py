"""CPU-only topology, rank-progress signatures and replica-aware accounting (vLLM port).

vLLM vocabulary instead of SGLang's: one process per (DP rank, TP rank); world = dp * tp. SGLang's
"DP attention" (attention replicated per DP rank, MoE spanning all ranks) is vLLM's data parallelism for a
MoE model: attention runs per DP rank, and the MoE layers span the whole DP x TP group, either as expert
parallelism (`--enable-expert-parallel`, EP = world) or tensor-parallel experts (moe_tp = world).
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeTopology:
    tp_size: int
    dp_size: int
    ep_size: int
    batch_size: int  # Global request concurrency, never per-DP concurrency.

    def __post_init__(self):
        if min(self.tp_size, self.dp_size, self.ep_size, self.batch_size) <= 0:
            raise ValueError("TP, DP, EP and global batch sizes must be positive")
        if self.ep_size not in (1, self.world_size):
            raise ValueError(f"vLLM expert parallelism spans the whole DP x TP group: ep-size must be 1 "
                             f"or {self.world_size}, got {self.ep_size}")
        if self.batch_size % self.dp_size:
            raise ValueError(f"Global batch size {self.batch_size} must divide evenly across {self.dp_size} DP ranks")

    @property
    def world_size(self):
        return self.tp_size * self.dp_size

    @property
    def enable_expert_parallel(self):
        return self.ep_size > 1

    @property
    def local_batch_size(self):
        return self.batch_size // self.dp_size

    @property
    def representative_ranks(self):
        """One process per DP replica (its TP rank 0); TP peers hold the same requests."""
        return [dp * self.tp_size for dp in range(self.dp_size)]

    def rank_state(self, rank):
        if not 0 <= rank < self.world_size:
            raise ValueError("Rank is outside the DP x TP process group")
        return dict(
            rank=rank, world_size=self.world_size,
            dp_rank=rank // self.tp_size, dp_size=self.dp_size,
            tp_rank=rank % self.tp_size, tp_size=self.tp_size,
            pp_rank=0, pp_size=1,
            moe_ep_size=self.ep_size, moe_tp_size=1 if self.enable_expert_parallel else self.world_size,
            enable_expert_parallel=self.enable_expert_parallel,
        )

    def request_id_offset(self, rank):
        return self.rank_state(rank)["dp_rank"] * self.local_batch_size

    def engine_overrides(self):
        """EngineArgs fields the bench owns (topology)."""
        return dict(tensor_parallel_size=self.tp_size, data_parallel_size=self.dp_size,
                    enable_expert_parallel=self.enable_expert_parallel,
                    distributed_executor_backend="external_launcher")

    def summary(self):
        return dict(tp_size=self.tp_size, dp_size=self.dp_size, ep_size=self.ep_size,
                    world_size=self.world_size, enable_expert_parallel=self.enable_expert_parallel,
                    global_batch_size=self.batch_size, local_batch_size=self.local_batch_size)


def compare_resolved_parallel(topology, rank, resolved):
    """Raise when the engine's resolved parallel config differs from the bench topology."""
    want = topology.rank_state(rank)
    mismatched = {key: (want[key], resolved.get(key)) for key in
                  ("dp_rank", "dp_size", "tp_rank", "tp_size", "pp_size", "enable_expert_parallel")
                  if key in resolved and resolved[key] != want[key]}
    if mismatched:
        raise ValueError(f"Resolved engine parallel state differs from the bench topology: {mismatched}")
    return want


_DIGEST_MOD = (1 << 61) - 1


def lens_digest(values):
    """Order-sensitive rolling hash; fits int64."""
    digest = 0
    for value in values:
        digest = (digest * 1000003 + int(value) + 1) % _DIGEST_MOD
    return digest


def rank_progress_signature(iteration, previous_lens, accept_lens, new_lens, final_lens, max_accept=6):
    """Never raise locally before the collective: encode invalid state instead.

    vLLM draws acceptance per request, so the accept VECTOR is digested (GLM compared one broadcast
    scalar). Every DP rank holds the same ISL multiset in the same order and the same per-request sampling
    seeds, so the vectors must agree across ranks elementwise.
    """
    if isinstance(final_lens, int):
        final_lens = [final_lens] * len(new_lens)
    sized = len(final_lens) == len(new_lens)
    valid = bool(previous_lens) and len(previous_lens) == len(accept_lens) == len(new_lens) and sized
    valid = valid and all(1 <= count <= max_accept for count in accept_lens)
    valid = valid and all(new == old + count for old, count, new in zip(previous_lens, accept_lens, new_lens))
    complete = sized and bool(new_lens) and all(new >= final for new, final in zip(new_lens, final_lens))
    return [int(valid), iteration, len(previous_lens), lens_digest(previous_lens),
            lens_digest(accept_lens), lens_digest(new_lens), int(complete)]


def validate_rank_progress(signatures):
    if not signatures or any(not row[0] or list(row) != list(signatures[0]) for row in signatures):
        raise ValueError(f"Cross-rank acceptance/progress/completion divergence: {signatures}")


def aggregate_rank_summaries(reports, topology):
    if len(reports) != topology.world_size or sorted(report["rank"] for report in reports) != list(range(topology.world_size)):
        raise ValueError("Expected exactly one report from every process")
    reports = sorted(reports, key=lambda report: report["rank"])
    first = reports[0]
    fields = ("complete", "verify_iterations", "batch_size", "input_lens", "input_len_uniform",
              "output_len", "emitted_per_request", "final_seq_lens", "raw_accept_tokens",
              "accept_histogram")
    for report in reports:
        if report["batch_size"] != topology.local_batch_size or any(report[key] != first[key] for key in fields):
            raise ValueError("Rank reports disagree on completion or acceptance exposure")
    replicas = [reports[rank] for rank in topology.representative_ranks]
    result = dict(first)
    result.update(topology.summary())
    result.update(batch_size=topology.batch_size, report_scope="global",
                  representative_ranks=topology.representative_ranks,
                  elapsed_seconds=max(report["elapsed_seconds"] for report in reports))
    for key in ("useful_output_tokens", "raw_accept_tokens", "num_correct_drafts", "terminal_extra_tokens"):
        result[key] = sum(report[key] for report in replicas)
    for key in ("emitted_per_request", "final_seq_lens"):
        result[key] = [value for report in replicas for value in report[key]]
    result["input_lens"] = list(first["input_lens"]) * len(replicas)
    histogram = Counter()
    for report in replicas:
        histogram.update({int(key): value for key, value in report["accept_histogram"].items()})
    result["accept_histogram"] = dict(sorted(histogram.items()))
    exposure = topology.batch_size * result["verify_iterations"]
    result["realized_accept_length"] = result["raw_accept_tokens"] / exposure if exposure else 0.0
    useful, elapsed = result["useful_output_tokens"], result["elapsed_seconds"]
    result["output_tokens_per_second"] = useful / elapsed if elapsed > 0 else 0.0
    result["output_tokens_per_second_per_gpu"] = result["output_tokens_per_second"] / topology.world_size
    result["effective_token_latency_ms_per_user"] = elapsed * 1000 * topology.batch_size / useful if useful else 0.0
    result["graph_execution_counts_scope"] = "rank_0"
    result["graph_execution_counts_by_rank"] = {str(r["rank"]): r["graph_execution_counts"] for r in reports}
    result["per_dp_useful_output_tokens"] = [report["useful_output_tokens"] for report in replicas]
    if "step_seconds_p50" in first:
        result["step_seconds_p50_by_rank"] = {str(r["rank"]): r["step_seconds_p50"] for r in reports}
    if "bootstrap_tokens_excluded" in first:
        result["bootstrap_tokens_excluded"] = sum(report["bootstrap_tokens_excluded"] for report in replicas)
    if "phase_seconds" in first:
        result["phase_seconds_max_rank"] = {key: max(report["phase_seconds"][key] for report in reports)
                                            for key in first["phase_seconds"]}
    result.pop("rank", None)
    result.pop("parallel_state", None)
    return result
