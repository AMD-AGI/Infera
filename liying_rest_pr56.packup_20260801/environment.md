# Environment

This run reused the cluster exactly as the predecessor kit left it, so the
hardware and fabric detail is not re-derived here — see
`glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial/environment.md`
(captured 2026-08-01 04:23 UTC, ~1 h before this run) and its
`env/env_chi2879.txt` / `env/env_chi2867.txt`. Recorded below is only what this
run depends on or changed.

## Nodes

| | chi2879 | chi2867 |
|---|---|---|
| role here | patch-in-place + all tests + build | build + verify only |
| data-plane IP | 10.2.122.10 | 10.2.122.44 |
| GPU | 8× MI355X **gfx950** | same |

No GPU work was done in this run: every gate is a unit/integration test or a
byte-level check. The live engines were only *read* (round 3), never restarted.

## Images

**Base** (pinned, unchanged from the predecessor):

    lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x
    RepoDigest sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d

**Reference, pre-existing** — `infera/engine-sglang:merged`, the ground truth
this run 对拍s against. **Not modified, not retagged, not removed**:

| node | image id |
|---|---|
| chi2879 | `sha256:1f7cf6964cee…` |
| chi2867 | `sha256:0d478433f1b3…` |

Its live containers `merged_run` (up 14 h at the start of this run) were used
for the in-place patch rounds. Their filesystems now carry the group-E patches;
**the image does not**. Restart the container from the image to get a clean one.

**Built by this run** — `infera/engine-sglang:merged-e`, from
`git archive` of `yihou.dev.glm52.merged.experiment` @ `b6819a6`:

| node | image id |
|---|---|
| chi2879 | `sha256:bfcb6462fa306743e0bf43b32ac0263ce9094e13591f6f748263e5348bf97e41` |
| chi2867 | `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |

Differing ids are expected — independent builds, so Rust objects and layer
timestamps differ. Check content equivalence, not digests.

> **The `-e` tag is deliberate.** Building over `:merged` would have destroyed
> the reference the CLAUDE.md ground-truth rule depends on. Two tags coexist.

## Repo state

| | |
|---|---|
| branch | `yihou.dev.glm52.merged.experiment` |
| worktree | `/home/yihou/dev/git.16-19/infera.glm52.merged` |
| HEAD at build | `b6819a6` (31 commits) |
| group E | `d3c0d6f`, `fd3540d`, `eef9bfc` |
| upstream compared against | PR #56 `llying/glm5p2_fp8_fixes`, 7 commits, refetched 2026-08-01, **unchanged and still OPEN** |

## Toolchains that matter

| | where | note |
|---|---|---|
| cargo | in-container `$HOME/.cargo/bin` | the **workstation's cargo 1.75 cannot be used** — `Cargo.lock` is v4 and it errors `lock file version 4 requires -Znext-lockfile-bump` |
| `LIBCLANG_PATH` | `/opt/rocm-7.2.0/lib/llvm/lib` | required by onig_sys/bindgen; the Dockerfile discovers it, an ad-hoc `cargo test` does not — see `notes.md` §2 |
| pytest (image) | 9.0.2, **no `pytest-asyncio`** | `@pytest.mark.asyncio` is an unknown mark there — see `notes.md` §3 |
| pytest (workstation) | with `pytest-asyncio` | which is why the same 4 e2e tests pass locally and fail in the image |

## External dependencies

Only one, and only for round 3 (which proved nothing and need not be repeated):

- Model / tokenizer: `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` — shared VAST NFS,
  mounted at `/mnt/vast` on both nodes. The Rust router needs it for
  `--kv-tokenizer-path`.

Rounds 1, 2, 4 and 5 need no model and no GPU.

## Secrets

None beyond cluster access:

- **Cluster SSH** — jump host `root@149.28.124.225`, then `ssh <node>`.
  Key-based; arrange your own. No registry login (all images are local), and
  neither etcd nor the router is authenticated on this cluster.
- **`gh`** for reading PR #56 — any token with read access to `AMD-AGI/Infera`.

No credential values appear anywhere in this kit.
