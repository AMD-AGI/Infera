# Assets deliberately NOT packed (size), and where to get them

Both are > 4 MB and both are reconstructible, so they are referenced by path
rather than copied in.

| asset | size | where it is | how to reobtain |
|---|---|---|---|
| engine image build context | 7.6 MB | `/shared_nfs/yihou_agentx_caseA/build/src.tar` | `git archive` of the branch's `deploy/docker` tree; or copy from the path |
| Case-A corpus tarball | 14 MB | `/shared_nfs/yihou_agentx_caseA/bench/caseA_conformance_corpus.tar.gz` | `git show pr173:scripts/AgentX_CaseA/caseA_conformance_corpus.tar.gz`, **or** regenerate byte-identically: `python3 gen_caseA_conformance.py corpus 200 42` (the generator IS packed, in this directory) |

The corpus is deterministic — seed 42 reproduces it byte-for-byte — so
`gen_caseA_conformance.py` + `verify_caseA.py` (both packed here) are sufficient
to rebuild and re-verify it without the tarball.
