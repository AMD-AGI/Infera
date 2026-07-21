# KV-Cache Management (kvd)

```{admonition} One-pager
:class: tip
The design of `infera.kvd` — the per-host daemon that builds a **tiered KV
cache** below the engine's GPU cache and **offloads / onboards** blocks across
GPU HBM → host RAM → NVMe → network. This page is the mechanism: the tiers, the
zero-copy transport, the on-disk format, and how it's tuned. For **how to turn it
on and the usage modes**, see [KV-Cache Offload](../features/kv_cache_offload.md).
```

```{graphviz}
digraph cache_tiers {
  rankdir=TB; bgcolor="transparent";
  node [shape=box style="rounded,filled" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.22,0.13"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10];

  L1 [fillcolor="#ffe3e3" color="#cc6666" label=<<b>L1 · GPU HBM</b> — engine prefix cache<br/><font color="#996666">fastest · smallest · not shared · lost on restart</font>>];
  L2 [fillcolor="#fff3cd" color="#caa300" label=<<b>L2 · host RAM shared arena</b> (kvd) — zero-copy mmap<br/><font color="#998400">cross-engine on one host · lost on restart</font>>];
  L3 [fillcolor="#e2efdd" color="#6a9a4a" label=<<b>L3 · local NVMe / NFS / WekaFS</b> (kvd)<br/><font color="#5e8048">durable · shared via mount</font>>];
  L4 [fillcolor="#dbe7ff" color="#5577cc" label=<<b>L4 · distributed</b> (kvd) — Mooncake / LMCache<br/><font color="#5b6f9e">durable · cross-node</font>>];

  L1 -> L2 [label="miss → ask kvd" penwidth=1.7 color="#5577cc"];
  L2 -> L3 [label="miss" penwidth=1.7 color="#5577cc"];
  L3 -> L4 [label="miss (only if L4 is the chosen --long-backend)" style=dashed color="#8a8a8a"];
  L1 -> L3 [label="GPU-direct (hipFile/AIS): engine reads/writes\lL3 chunk files directly, bypassing the daemon\l" style=dashed color="#8a8a8a" constraint=false];
}
```

Higher tiers are faster and smaller; a miss falls through to the next tier down.
**L1** is the engine's own GPU cache; **L2–L4** are kvd's. The **long tier is one
`--long-backend`** — local `tablespace` (L3) **or** a distributed backend (L4),
never both.

The daemon owns L2 (the RAM arena) and, with GPU-direct **off**, also does the L3
disk I/O via its tablespace writer. With **GPU-direct on** (`hipFile`/AIS) the
engine reads/writes the L3 chunk files **directly** (DMA straight into GPU
memory); those files are connector-owned, so the daemon's `--long-bytes` doesn't
bound them — the connector caps that tier via `INFERA_KVD_L3_BUDGET_BYTES` /
`INFERA_KVD_L3_FREE_FLOOR`.

## Why a separate daemon

The engine's own KV cache (L1) lives in GPU memory: it's fast, but small, not
shared between engines, and gone on restart. `kvd` adds the tiers below it:

| Tier | Storage | Managed by | Restart-durable | Shared |
|---|---|---|:---:|---|
| **L1** | GPU HBM | the engine (vLLM/SGLang) | No | No |
| **L2** | host pinned RAM (memfd shared arena) | kvd | No | cross-engine, one host, zero-copy |
| **L3** | local NVMe / NFS / WekaFS | kvd | **Yes** | via the shared mount |
| **L4** | distributed (Mooncake / Redis / S3 / NIXL / WEKA) | kvd | **Yes** | cross-node |

### What fires when — a GET

```{graphviz}
digraph tier_lookup {
  rankdir=TB; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.2,0.12"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10];

  R  [label="Request"];
  L1 [shape=diamond style=filled fillcolor="#fff3cd" color="#caa300" label="L1 GPU\ncache?"];
  U  [label="use it" fillcolor="#e2efdd" color="#6a9a4a"];
  K  [shape=diamond style=filled fillcolor="#fff3cd" color="#caa300" label="ask kvd"];
  M  [label="CopyFree mmap read" fillcolor="#e2efdd" color="#6a9a4a"];
  P  [label="load → promote to L2" fillcolor="#e2efdd" color="#6a9a4a"];
  C  [label="engine computes prefill\n→ kvd stores it"];

  R -> L1;
  L1 -> U [label="hit"];
  L1 -> K [label="miss"];
  K -> M [label="L2 hit (RAM)"];
  K -> P [label="L3 hit (disk)"];
  K -> C [label="miss"];
}
```

A SET flows the other way: new blocks land in L2; eviction from L2 writes back to
the long tier (L3/L4), unless the request marked them `ephemeral`. These
**offloads (and the load-backs) are asynchronous** — they run off the
token-generation path, so spilling a block to disk never stalls decode, and kvd
batches the writes for throughput.

## The zero-copy win (L2)

The RAM tier is a memfd-backed pinned **shared arena**. When a vLLM/SGLang
worker opts in at the UDS handshake, kvd passes the arena's file descriptor over
`SCM_RIGHTS` and the engine `mmap`s the same region. A `get()` then returns just
`(slot_offset, length, version)` — the engine reads bytes straight from its own
mmap. No socket body, no `bytes(...)` copy. The gain vs the legacy inline-bytes
path is largest on NFS-backed L3 (prefetch warms the arena; the bytes path can't
benefit).

## L3 on-disk design — the tablespace

The L3 disk tier never writes one OS file per cached block. Instead it
pre-allocates a **small, fixed set of large "container" files** and packs blocks
into fixed-size **slots** inside them — the same pattern databases use for a
*tablespace* (and 3FS for its chunk engine). The file count is a constant
(`max_bytes / container_bytes`), so a 64 GB region with 1 GB containers is **64
files no matter how many blocks it holds** — which is what keeps filesystem
metadata (inode lookup, dentry cache, the restart scan) from becoming the
bottleneck the way one-file-per-block did past ~100K entries, especially on NFS.

```{graphviz}
digraph tablespace_layout {
  rankdir=TB;
  bgcolor="transparent";
  node [fontname="Helvetica,Arial,sans-serif", fontsize=11];
  edge [fontname="Helvetica,Arial,sans-serif", fontsize=10, color="#8a8a8a"];

  subgraph cluster_region {
    label=<<b>&lt;region_dir&gt;/</b> &#8212; file count = max_bytes / container_bytes (constant)>;
    labeljust="l";
    fontname="Helvetica,Arial,sans-serif"; fontsize=11; fontcolor="#444444";
    style="rounded"; color="#cccccc"; margin=14;

    c0 [shape=plaintext margin=0 label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8" COLOR="#9fb6cf">
        <TR><TD COLSPAN="4" BGCOLOR="#eef2f7" ALIGN="LEFT"><b>containers/0000.bin</b>  ·  pre-allocated, fixed size</TD></TR>
        <TR>
          <TD BGCOLOR="#cdeccd">slot 0<BR/><b>block A</b></TD>
          <TD BGCOLOR="#f4f4f4"><FONT COLOR="#9a9a9a">slot 1<BR/>free</FONT></TD>
          <TD BGCOLOR="#cdeccd">slot 2<BR/><b>block B</b></TD>
          <TD BGCOLOR="#f4f4f4"><FONT COLOR="#9a9a9a">slot 3<BR/>free</FONT></TD>
        </TR>
      </TABLE>>];

    c1 [shape=plaintext margin=0 label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8" COLOR="#9fb6cf">
        <TR><TD COLSPAN="3" BGCOLOR="#eef2f7" ALIGN="LEFT"><b>containers/0001.bin</b></TD></TR>
        <TR>
          <TD BGCOLOR="#cdeccd">slot 0<BR/><b>block C</b></TD>
          <TD BGCOLOR="#f4f4f4"><FONT COLOR="#9a9a9a">slot 1<BR/>free</FONT></TD>
          <TD BGCOLOR="#f4f4f4"><FONT COLOR="#9a9a9a">&#8230;</FONT></TD>
        </TR>
      </TABLE>>];

    log  [shape=note style=filled fillcolor="#dbe7ff" color="#5577cc"
          label="index.log\nappend-only journal\n(fsync per PUT)"];
    snap [shape=note style=filled fillcolor="#dbe7ff" color="#5577cc"
          label="index.snapshot.json (+ .bak)\nperiodic compacted index"];

    c0 -> c1 [style=invis];
    c1 -> log [style=invis];
    log -> snap [style=invis];
  }

  bit [shape=box style="rounded,filled" fillcolor="#fff3cd" color="#caa300"
       label="RAM\nfree-slot bitset  +  key → (container, slot) index"];

  bit -> c0   [label="  allocates /\l  locates slots\l" style=dashed];
  bit -> c1   [style=dashed];
  snap -> bit [label="  startup: load snapshot…" style=bold color="#5577cc" fontcolor="#5577cc"];
  log  -> bit [label="  …then replay journal tail" style=bold color="#5577cc" fontcolor="#5577cc"];
}
```

- **Slots.** Each container is carved into fixed `--tablespace-slot-bytes` slots
  (default 64 KB, sized to the typical packed-KV block). A value larger than one
  slot is rejected (the caller recomputes) — slots are not chained.
- **Allocator.** A bitset in RAM tracks free slots; allocation and the
  `key → (container, slot)` index are in-memory. **Disk durability is the
  journal**, not the bitset.

### Write path & crash safety

Every PUT is **slot-reserve → write+fsync data → append+fsync journal**, so a
crash can never surface a half-written block:

```{graphviz}
digraph write_path {
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc"
        fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.18,0.12"];
  edge [color="#5577cc" penwidth=1.4 arrowsize=0.8];

  r [label="1\nreserve slot\n(RAM bitset)"];
  w [label="2\npwrite value at\nslot offset + fsync"];
  j [label="3\nappend PUT to\nindex.log + fsync"];
  i [label="4\nupdate in-memory\nindex"];
  r -> w -> j -> i;
}
```

A crash **between steps 2 and 3** leaves a slot written on disk but with no
journal record — on restart it's simply seen as free and overwritten later, never
served as a phantom block. Steps 3 and 4 are atomic from a restart's point of view
(step 4 is RAM-only). On restart kvd loads `index.snapshot.json` (written atomically on graceful
shutdown / periodic compaction), then replays any journal entries appended since
the snapshot — so both a clean shutdown **and** a hard kill recover the full
index. The journal is plain JSON on purpose: `cat index.log` shows the operation
history, and a parse failure is obvious (vs. an opaque embedded-DB corruption).

## Storage-aware auto-tuning

kvd inspects the backing device at startup so you don't hand-tune per storage
class:

- `--io-mode auto` (default) — O_DIRECT on NVMe, buffered on NFS/SATA.
- IO concurrency is selected internally by device class (no CLI flag); the L3
  writer is plain `pwrite`/O_DIRECT — no extra dependencies to build or ship.
  For the striped region you can cap intra-shard parallelism with
  `--long-workers-per-shard` (default `8`).
- Low NFS `wsize` → a startup WARN with the exact remount command.

## Sizing principles

- **L2 (`--max-bytes`)** — as much host RAM as you can spare for KV; this is the
  hottest, zero-copy tier. On a big-RAM host (~3 TB/host on this fleet) L2 can be
  large.
- **L3 (`--long-bytes`)** — the working-set's durable footprint; sized to hold the
  prefixes you want to survive restarts.
- **Grow capacity as you go down** — each tier should be **≥ the one above**
  (L3 ≥ L2 ≥ the engine's GPU L1). A lower tier smaller than the one above it just
  churns: blocks evict before they're ever reused.

## Verify it's running

After starting the daemon and at least one engine, send the same prompt twice
(or any prompt that shares a long prefix), then check the daemon counters:

```bash
python -m infera.kvd.statctl --socket /var/run/infera-kvd.sock
# healthy output: sets_total > 0, gets_total > 0, hits_total > 0
```

`hits_total == 0` after a repeated prompt usually means the engine isn't
connected to the daemon — confirm `INFERA_KVD_SOCKET` is set and points to the
same socket path. Under GPU-direct the daemon counters stay 0 by design (the
connector owns the files directly); use the engine's `External prefix cache hit
rate` log line and check for `.kvcache` files under `INFERA_KVD_HIPFILE_ROOTS`
instead.

## Related

- [KV-Cache Offload](../features/kv_cache_offload.md) — how to turn it on and the
  usage modes (RAM / disk / distributed / GPU-direct, per-request retention).
- [CLI reference](../reference/cli.md) — the daemon flags and connector env vars.

