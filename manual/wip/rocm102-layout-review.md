# Infera 0.3.3 documentation layout review

Working review checklist for the ROCm 10.2 documentation Epic. This file is
excluded from the published manual by `wip/**` in `conf.py`.

## Confirmed inputs

- Target release: Infera 0.3.3. ROCm 10.2 validation is confirmed complete.
- Source inspected: main at d1886ebe64fbdd6d8ba9fa7de3e290ac155a1792.
- GitHub latest release checked: v0.3.3. Its ROCm 10.2 Dockerfile exists.
- The current matrix / Dockerfile still record a 10.2.0rc0 image and a
  disaggregated-decode defect. Those specific facts are retained pending a
  replacement image or explicit resolution evidence; overall validation does
  not imply the defect is resolved.

## Shared files and site entry points

| Item | Gap and change |
| --- | --- |
| `conf.py` | Replace hardcoded 0.1 with 0.3.3; share version through MyST substitutions. Include level-four headings in cross-reference discovery. |
| `sphinx/_toc.yml.in` | Replace eight groups with Install, Conceptual, Reference, Examples, About; include every existing published page once. |
| `index.md` | Make installation the first entry card. Keep the root URL and page content. |
| `README.md` | Correct theme description and document navigation/version maintenance. |
| AI Ecosystem sidebar | Currently links to the Infera root; update in the owning site to the install URL below. Not changed in this repository. |
| AI Ecosystem landing page | Same root link and required destination as sidebar. Not changed in this repository. |

Required AI Ecosystem destination:
`https://rocm.docs.amd.com/projects/infera/en/latest/getting_started/installation.html`

## Page-by-page gap list

| Source below `manual/` | Previous → target group | Change |
| --- | --- | --- |
| `getting_started/installation.md` | Getting started → Install | Add release version, prerequisites, ROCm / method tabs, version-specific installation links, pinned commands, and package version check. Preserve old anchors. |
| `getting_started/quickstart.md` | Getting started → Install | Replace old 0.1.1 image example with locally built 0.3.3 image; distinguish the 7.2 Qwen example from 10.2 validation. |
| `serving/deployment.md` | Serving → Install | Navigation only; content and URL retained. |
| `serving/kubernetes.md` | Serving → Install | Navigation only; content and URL retained. |
| `getting_started/overview.md` | Getting started → Conceptual | Navigation only; content and URL retained. |
| `components/server.md` | Components → Conceptual | Navigation only; content and URL retained. |
| `components/engines.md` | Components → Conceptual | Navigation only; content and URL retained. |
| `components/operator.md` | Components → Conceptual | Navigation only; content and URL retained. |
| `components/kvd.md` | Components → Conceptual | Navigation only; content and URL retained. |
| `features/pd_disaggregation.md` | Features → Conceptual | Navigation only; content and URL retained. |
| `features/kv_cache_offload.md` | Features → Conceptual | Navigation only; content and URL retained. |
| `features/kv_aware_routing.md` | Features → Conceptual | Navigation only; content and URL retained. |
| `features/routing_and_transport.md` | Features → Conceptual | Navigation only; content and URL retained. |
| `features/scaling.md` | Features → Conceptual | Navigation only; content and URL retained. |
| `features/sla_planner.md` | Features → Conceptual | Navigation only; content and URL retained. |
| `features/graceful_shutdown.md` | Features → Conceptual | Navigation only; content and URL retained. |
| `simulation/overview.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/projection_runs.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/simulation_runs.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/anchors.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/workloads.md` | Simulation → Conceptual | Use a supported lexer for the CSV example to allow a warning-free build. |
| `simulation/fleet_and_routing.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/disaggregation.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/sweeps.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/tuning_agent.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/modeled_features.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/environment.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `simulation/boundaries.md` | Simulation → Conceptual | Navigation only; content and URL retained. |
| `features/feature_matrix.md` | Features → Reference | Navigation only; content and URL retained. |
| `features/compatibility_matrix.md` | Features → Reference | Use shared release version; identify 10.2 validation while preserving rc0 provenance, known defect, limits, and old section anchors. |
| `reference/cli.md` | Reference → Reference | Navigation only; content and URL retained. |
| `reference/environment.md` | Reference → Reference | Navigation only; content and URL retained. |
| `reference/benchmarking.md` | Reference → Reference | Navigation only; content and URL retained. |
| `reference/preflight.md` | Reference → Reference | Navigation only; content and URL retained. |
| `reference/troubleshooting.md` | Reference → Reference | Navigation only; content and URL retained. |
| `reference/glossary.md` | Getting started → Reference | Navigation only; content and URL retained. |
| `recipes/index.md` | Recipes → Examples | Navigation only; content and URL retained. |
| `recipes/glm5.2.md` | Recipes → Examples | Navigation only; content and URL retained. |
| `recipes/glm5.2-fp8-gfx942.md` | Recipes → Examples | Navigation only; content and URL retained. |
| `recipes/kimi-k3.md` | Recipes → Examples | Navigation only; content and URL retained. |
| `recipes/kimi-k3-optimized.md` | Recipes → Examples | Navigation only; content and URL retained. |
| `examples/k8s_kimi_k3.md` | Examples → Examples | Navigation only; content and URL retained. |
| `examples/k8s_pd_1p1d_mooncake.md` | Examples → Examples | Navigation only; content and URL retained. |
| `examples/k8s_glm5.2_sglang.md` | Examples → Examples | Navigation only; content and URL retained. |
| `examples/k8s_glm5.3.md` | Examples → Examples | Repair two source-tree links as release-pinned GitHub links. |
| `examples/sglang_1p2d_kimi2.6.md` | Examples → Examples | Navigation only; content and URL retained. |
| `examples/sglang_mix_dsv4.md` | Examples → Examples | Navigation only; content and URL retained. |
| `examples/sglang_1p1d_dsv4.md` | Examples → Examples | Navigation only; content and URL retained. |
| `examples/atom_4p4d_glm5.2.md` | Examples → Examples | Navigation only; content and URL retained. |
| `about/release-notes.md` | New → About | Add documentation release notes; no unverified software-fix claims. |

## Exceptions and publication dependencies

- Keep existing source directories / URLs while changing navigation groups.
  This preserves both the Infera root and individual page links; it does not
  redirect the Infera root to installation. The home page instead provides an
  explicit Install card.
- ROCm 10.3 is an alpha preview: retain exact container provenance instead of
  supplying an unverified public release installation command.
- The expected ROCm 10.2 public install URL returns HTTP 404:
  `https://rocm.docs.amd.com/en/docs-10.2.0/install/rocm.html`.
  The install page explicitly documents the container path and the missing
  public guide instead of publishing a broken link. Add the matching guide
  when available; this is an exception to the Install ROCm link requirement.
- Cross-check the final version and validation scope against ROCM-33743
  during content review; this change retains the image-level limitations
  recorded in the existing matrix.
- SME name, agreement on this gap list, Quality ticket, and QA signoff are not
  recorded here as completed. Add the references before docs publication.
- The ROCm 10.2 release blog URL was not supplied. Its existing Infera root
  destination remains available on the live site; verify the actual blog link
  and deployed result before publication.
- Release notes describe documentation changes only. Product release highlights,
  fixed issues, and replacement validation artifacts require owner-supplied facts.

## Validation

- Strict HTML build passed with the pinned documentation dependencies on Python 3.12:
  `python -m sphinx -b html -W --keep-going -n manual manual/_build/html`.
- All 49 existing navigation pages are retained exactly once; one release-notes
  page is added. The five target navigation groups are present.
- Rendered checks passed for version substitutions, both selector groups,
  legacy anchors, and 4,504 local links. Deployment-relative paths on the
  generated 404 page were resolved against the site root. OneTrust cookie
  controls are JavaScript actions, excluded from static anchor validation.
- Seven checked public destinations returned HTTP 200. The expected ROCm 10.2
  install destination returned HTTP 404 and is not linked from published content.

Local validation does not constitute deployment, an AI Ecosystem site update,
QA signoff, or agreement on the gap list.
