"""CPU-only topology, DP-forward metadata and replica-aware accounting.

Port of the GLM-5.2 tool's topology.py. Unchanged: DecodeTopology, lens_digest, the 7-int rank
progress signature, validate_rank_progress, aggregate_rank_summaries. Adapted:

- ``parallel_state_kwargs`` is no longer handed to a ``ParallelState`` constructor (sglang
  6be0d75ae builds the parallel state inside ``bootstrap.init_parallel_runtime``); the driver
  compares it field by field with the engine's resolved ``get_parallel()`` instead.
- ``prepare_dp_metadata`` sets the 6be0d75ae ScheduleBatch field names (``can_run_decode_cuda_graph``
  replaced ``can_run_dp_cuda_graph``; draft / prefill graph votes are separate fields). With
  ``sync`` given it runs the engine's own MLP-sync gather (``prepare_mlp_sync_batch_raw``, the
  function the Scheduler and ``bench_one_batch`` call) and then asserts the gathered counts.
- ``rank_progress_signature`` takes the accept bound (verify window) as a parameter.
"""
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


# Engine attribute names (sglang 6be0d75ae get_parallel()) for the kwargs above; dp_rank / gpu_id
# are launcher-side and not compared.
_PARALLEL_FIELDS = ("tp_rank", "tp_size", "pp_rank", "pp_size", "attn_tp_rank", "attn_tp_size",
                    "attn_cp_rank", "attn_cp_size", "attn_dp_rank", "attn_dp_size",
                    "moe_ep_rank", "moe_ep_size", "moe_dp_rank", "moe_dp_size")


def compare_parallel_state(topology, rank, resolved):
    """Return {field: (expected, resolved)} for every mismatch; missing engine attributes are skipped."""
    expected = topology.parallel_state_kwargs(rank)
    mismatches = {}
    for name in _PARALLEL_FIELDS:
        if not hasattr(resolved, name):
            continue
        value = getattr(resolved, name)
        if value != expected[name]:
            mismatches[name] = (expected[name], value)
    return mismatches


def expected_decode_global_tokens(topology, gathered_len):
    """Gathered decode counts are BASE requests: [local]*dp with a TP gather, [local] without."""
    if gathered_len == 1:
        return [topology.local_batch_size]
    return [topology.local_batch_size] * topology.dp_size


def prepare_dp_metadata(batch, topology, *, is_extend, disable_cuda_graph, sync=None):
    """Set the DP / graph-gating metadata the forward path reads.

    ``sync`` (on GPU): the engine's MLP-sync gather; then the gathered decode counts are asserted
    against the topology. ``sync=None`` (CPU tests, ``--dp-metadata static``): the fields are set
    by hand from the topology, exactly as the GLM-5.2 tool did, with 6be0d75ae field names.
    """
    if sync is not None:
        sync(batch)
        if not is_extend:
            got = list(batch.global_num_tokens or [])
            want = expected_decode_global_tokens(topology, len(got))
            if got != want:
                raise ValueError(f"MLP-sync gathered global_num_tokens={got}, expected {want}")
        return batch
    tokens = getattr(batch, "extend_num_tokens", None) if is_extend else None
    local = tokens if tokens is not None else topology.local_batch_size
    batch.global_num_tokens = [local] * topology.dp_size
    batch.global_num_tokens_for_logprob = [topology.local_batch_size] * topology.dp_size
    batch.is_extend_in_batch = is_extend
    batch.can_run_decode_cuda_graph = not is_extend and not disable_cuda_graph
    batch.can_run_dp_draft_cuda_graph = not disable_cuda_graph
    batch.can_run_dp_prefill_cuda_graph = False
    batch.dp_prefill_cuda_graph_max_prefix_len = 0
    return batch


_DIGEST_MOD = (1 << 61) - 1


def lens_digest(values):
    """Order-sensitive rolling hash; fits int64. Divergence in ANY position or in
    ordering changes the digest, which element-0 comparison could not detect."""
    digest = 0
    for value in values:
        digest = (digest * 1000003 + int(value) + 1) % _DIGEST_MOD
    return digest


def rank_progress_signature(iteration, previous_lens, accept_lens, new_lens, final_lens, max_accept=6):
    """Never raise locally before the collective: encode invalid state instead."""
    if isinstance(final_lens, int):
        final_lens = [final_lens] * len(new_lens)  # uniform-ISL callers (compare_server.py)
    sized = len(final_lens) == len(new_lens)
    valid = bool(previous_lens) and len(previous_lens) == len(accept_lens) == len(new_lens) and sized
    # Acceptance stays uniform by upstream construction (one scalar drawn per iteration and
    # broadcast); sequence lengths do not, so they are covered by a digest instead.
    valid = valid and len(set(accept_lens)) == 1
    valid = valid and all(1 <= count <= max_accept for count in accept_lens)
    valid = valid and all(new == old + count for old, count, new in zip(previous_lens, accept_lens, new_lens))
    # Each request finishes at its own isl_i + output_len, so completion is elementwise.
    complete = sized and bool(new_lens) and all(new >= final for new, final in zip(new_lens, final_lens))
    return [int(valid), iteration, len(previous_lens), lens_digest(previous_lens),
            accept_lens[0] if accept_lens else -1, lens_digest(new_lens), int(complete)]


def validate_rank_progress(signatures):
    if not signatures or any(not row[0] or list(row) != list(signatures[0]) for row in signatures):
        raise ValueError(f"Cross-rank acceptance/progress/completion divergence: {signatures}")


def aggregate_rank_summaries(reports, topology):
    if len(reports) != topology.tp_size or sorted(report["rank"] for report in reports) != list(range(topology.tp_size)):
        raise ValueError("Expected exactly one report from every TP process")
    reports = sorted(reports, key=lambda report: report["rank"])
    first = reports[0]
    # Uniform shared coins keep every process in the same collective sequence.
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
    # Every replica holds the same vector (asserted above), so repeating it aligns elementwise
    # with the concatenations and keeps len(input_lens) == len(final_seq_lens) == batch_size.
    result["input_lens"] = list(first["input_lens"]) * len(replicas)
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
