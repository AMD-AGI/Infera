"""CPU-only topology, DP-forward metadata and replica-aware accounting."""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeTopology:
    tp_size: int
    ep_size: int
    batch_size: int  # Global request concurrency, never per-DP concurrency.
    enable_dp_attention: bool = False

    def __post_init__(self):
        if min(self.tp_size, self.ep_size, self.batch_size) <= 0:
            raise ValueError("TP, EP and global batch sizes must be positive")
        if self.tp_size % self.ep_size:
            raise ValueError("EP size must divide TP size")
        if self.batch_size % self.dp_size:
            raise ValueError("Global batch size must divide evenly across attention DP shards")

    @property
    def dp_size(self):
        return self.tp_size if self.enable_dp_attention else 1

    @property
    def local_batch_size(self):
        return self.batch_size // self.dp_size

    @property
    def representative_ranks(self):
        return list(range(self.tp_size)) if self.enable_dp_attention else [0]

    def parallel_state_kwargs(self, rank):
        if not 0 <= rank < self.tp_size:
            raise ValueError("Rank is outside the TP process group")
        return dict(
            tp_rank=rank, tp_size=self.tp_size, pp_rank=0, pp_size=1,
            dp_rank=rank if self.enable_dp_attention else None, dp_size=self.dp_size,
            attn_tp_rank=0 if self.enable_dp_attention else rank,
            attn_tp_size=self.tp_size // self.dp_size,
            attn_cp_rank=0, attn_cp_size=1, attn_dcp_rank=0, attn_dcp_size=1,
            attn_dp_rank=rank if self.enable_dp_attention else 0, attn_dp_size=self.dp_size,
            moe_ep_rank=rank // (self.tp_size // self.ep_size), moe_ep_size=self.ep_size,
            moe_dp_rank=0, moe_dp_size=1, gpu_id=rank,
        )

    def request_id_offset(self, rank):
        return self.parallel_state_kwargs(rank)["attn_dp_rank"] * self.local_batch_size

    def summary(self):
        return dict(tp_size=self.tp_size, ep_size=self.ep_size, dp_size=self.dp_size,
                    enable_dp_attention=self.enable_dp_attention,
                    global_batch_size=self.batch_size, local_batch_size=self.local_batch_size)


def prepare_dp_metadata(batch, topology, *, is_extend, disable_cuda_graph):
    # ScheduleBatch counts are BASE requests. The pinned ForwardBatch scales them
    # using spec_info separately for draft, target verify and draft extension.
    batch.global_num_tokens = [topology.local_batch_size] * topology.dp_size
    batch.global_num_tokens_for_logprob = batch.global_num_tokens[:]
    batch.is_extend_in_batch = is_extend
    batch.can_run_dp_cuda_graph = not is_extend and not disable_cuda_graph
    batch.can_run_dp_breakable_cuda_graph = False


def rank_progress_signature(iteration, previous_lens, accept_lens, new_lens, final_len):
    """Never raise locally before the collective: encode invalid state instead."""
    valid = bool(previous_lens) and len(previous_lens) == len(accept_lens) == len(new_lens)
    valid = valid and len(set(accept_lens)) == 1 and len(set(previous_lens)) == 1
    valid = valid and all(1 <= count <= 6 for count in accept_lens)
    valid = valid and all(new == old + count for old, count, new in zip(previous_lens, accept_lens, new_lens))
    return [int(valid), iteration, len(previous_lens), previous_lens[0] if previous_lens else -1,
            accept_lens[0] if accept_lens else -1, new_lens[0] if new_lens else -1,
            int(bool(new_lens) and all(length >= final_len for length in new_lens))]


def validate_rank_progress(signatures):
    if not signatures or any(not row[0] or list(row) != list(signatures[0]) for row in signatures):
        raise ValueError(f"Cross-rank acceptance/progress/completion divergence: {signatures}")


def aggregate_rank_summaries(reports, topology):
    if len(reports) != topology.tp_size or sorted(report["rank"] for report in reports) != list(range(topology.tp_size)):
        raise ValueError("Expected exactly one report from every TP process")
    reports = sorted(reports, key=lambda report: report["rank"])
    first = reports[0]
    # Uniform shared coins keep every process in the same collective sequence.
    fields = ("complete", "verify_iterations", "batch_size", "input_len", "output_len",
              "emitted_per_request", "final_seq_lens", "raw_accept_tokens", "accept_histogram")
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
    histogram = Counter()
    for report in replicas:
        histogram.update({int(key): value for key, value in report["accept_histogram"].items()})
    result["accept_histogram"] = dict(sorted(histogram.items()))
    exposure = topology.batch_size * result["verify_iterations"]
    result["realized_accept_length"] = result["raw_accept_tokens"] / exposure if exposure else 0.0
    useful, elapsed = result["useful_output_tokens"], result["elapsed_seconds"]
    result["output_tokens_per_second"] = useful / elapsed if elapsed > 0 else 0.0
    result["output_tokens_per_second_per_gpu"] = result["output_tokens_per_second"] / topology.tp_size
    result["effective_token_latency_ms_per_user"] = elapsed * 1000 * topology.batch_size / useful if useful else 0.0
    result["graph_execution_counts_scope"] = "rank_0"
    result["graph_execution_counts_by_rank"] = {str(report["rank"]): report["graph_execution_counts"] for report in reports}
    result["per_dp_useful_output_tokens"] = [report["useful_output_tokens"] for report in replicas]
    if "bootstrap_tokens_excluded" in first:
        result["bootstrap_tokens_excluded"] = sum(report["bootstrap_tokens_excluded"] for report in replicas)
    if "phase_seconds" in first:
        result["phase_seconds_max_rank"] = {key: max(report["phase_seconds"][key] for report in reports) for key in first["phase_seconds"]}
    result.pop("rank", None)
    result.pop("parallel_state", None)
    return result
