"""Rebuild readable records from preserved payloads; never run a benchmark.

Run with the repository's agent_sys on PYTHONPATH. Published payloads are checked
against the import manifest before any generated JSON is written.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from kernel_registry.registry import Registry, file_record

ROOT = Path(__file__).resolve().parent
PAYLOAD_IDENTITIES = {
    item["path"]: item for item in json.loads((ROOT / "payload_manifest.json").read_text())
}
REVALIDATION = (
    "Published payloads and adapters are redacted historical derivatives. Historical correctness and "
    "performance apply to the original bytes; revalidate the published versions on the target GPU stack."
)
TASKS = {
    "sparse_mla": ("Sparse MLA decode", "sparse_mla_kernel", 14, "SPARSE"),
    "indexer": ("DSA indexer logits and top-k", "dsa_indexer_kernel", 15, "INDEXER"),
    "moe_mxfp4": ("MXFP4 routed MoE", "moe_mxfp4_kernel", 14, "MOE"),
    "dcp_sparse": ("DCP4 sparse MLA split-K", "dcp_sparse_kernel", 5, "DCP_SPARSE"),
    "dcp_indexer": ("DCP4 split indexer", "dcp_indexer_kernel", 4, "DCP_INDEXER"),
    "fused_indexer": ("gfx950 fused decode indexer", "fused_indexer_kernel", 4, "FUSED_INDEXER"),
}
KF_PACK = "payload/evidence/glm52.kf-decode-kernels-e2e.packup_20261023"
DCP_PACK = "payload/evidence/glm52.kf-dcp-and-fused-indexer.packup_20261025"
PD_PACK = "payload/evidence/glm52.llrepro-kf-ar-topk.packup_20261024"
PD_FINAL = "payload/evidence/glm52.llrepro-garblefree-sweep.packup_20261028"
GAPS = [
    "Full benchmark-time package inventory",
    "Host driver/OS/Python capture",
    "Timestamp timezone was not recorded",
    "External input availability",
]


def read(path):
    return json.loads((ROOT / path).read_text())


def file(path, role=None):
    result = file_record(ROOT, str(path))
    identity = PAYLOAD_IDENTITIES.get(str(path))
    if identity:
        result["representation"] = identity["representation"]
        result["historical_original_sha256"] = identity["historical_original_sha256"]
    if role:
        result["role"] = role
    return result


def files(paths):
    return [file(p.relative_to(ROOT) if isinstance(p, Path) else p) for p in paths]


def write(path, document):
    document["source_bytes"] = "redacted"
    document.setdefault("provenance", {}).update(
        representation="published redacted historical derivative",
        gpu_revalidated=False,
    )
    if document["kind"] in {"kernel", "run"}:
        limitations = document.setdefault("limitations", [])
        if REVALIDATION not in limitations:
            limitations.append(REVALIDATION)
    if document["kind"] == "run":
        document["correctness"]["applies_to"] = "historical originals"
    if document["kind"] == "kernel":
        for version in document["versions"]:
            version.setdefault("provenance", {})["source_bytes"] = "redacted"
        document["integration"]["validation_status"] = "target GPU revalidation required"
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def verify_payload():
    for item in read("payload_manifest.json"):
        actual = file(item["path"])
        if (actual["sha256"], actual["size"]) != (item["sha256"], item["size_bytes"]):
            raise ValueError(f"Changed preserved payload: {item['path']}")


def environment(name, image=None, digest=None, evidence=(), packages=None) -> dict[str, Any]:
    return dict(
        kind="environment",
        name=name,
        description="Historical environment imported from retained evidence; not captured anew.",
        container=dict(image=image, digest=digest),
        hardware=dict(device="AMD Instinct MI355X", architecture="gfx950", count=None, driver=None),
        system=dict(os=None, kernel=None, python=None),
        packages=packages or {},
        repositories=[],
        evidence=files(evidence),
        missing=GAPS[:],
        provenance=dict(import_method="Archived task and runtime records", rerun=False),
    )


def make_environments():
    workset = environment("kf-workset", evidence=["payload/tasks/sparse_mla/program.md"])
    workset["missing"].append("Per-campaign immutable image identity and software stack")
    write("environments/kf-workset.json", workset)
    e2e = environment(
        "historical-stack-a",
        "historical-sglang-image-a",
        "sha256:5b6565ef3e4e7dcca02423d509f842a74cf7416b18ed94edbf12e2be100f7376",
        [
            f"{KF_PACK}/environment.md",
            f"{KF_PACK}/cluster/v0521dbg_20261010/runs/kfb2/agentx-c56-A1/runtime.env",
        ],
        {
            "torch": "2.9.1+rocm7.2.0",
            "triton": "3.7.0+amd.rocm7.2.0.git89002410",
            "amd-aiter": "0.1.25.dev178",
            "flydsl": "0.3.4.1",
            "sglang": "0.5.22.dev20261020+geae134d61f",
        },
    )
    e2e["repositories"] = [
        dict(url="https://github.com/sgl-project/sglang", commit="93034eba91", dirty=None),
        dict(url="https://github.com/ROCm/aiter", commit="2bb549102", dirty=None),
        dict(
            url="https://github.com/SemiAnalysisAI/InferenceX",
            commit="918524ff94045b3f091115f1051c22a8588edf2b",
            dirty=False,
        ),
    ]
    write("environments/historical-stack-a.json", e2e)
    ppda = environment(
        "historical-stack-c",
        "historical-sglang-image-c",
        "sha256:1b78f7ccce06ef9fce39818a6d6536213db290f64fb5864eeb269956f89f27e6",
        ["payload/evidence/glm52.ppda-pp4-dpa4-kf.packup_20261023/results/rounds/ppda2/audit.txt"],
    )
    write("environments/historical-stack-c.json", ppda)
    pd = environment(
        "historical-stack-b",
        "historical-sglang-image-b",
        "sha256:f69f49e39cbf97e933c742f61d0d9a3c45914e51ab01c99ac4b6eb859858a967",
        [
            "payload/evidence/pd-image/image_id.txt",
            "payload/evidence/pd-image/build_ll.registry.sh",
            f"{PD_PACK}/environment.md",
        ],
        {"sglang": "0.5.22.dev20261018+g8dee3f6bdb"},
    )
    write("environments/historical-stack-b.json", pd)


def make_kernels():
    kernels, selected = {}, {}
    external = read("external_fixtures.json")
    for name, (description, basename, integrated_iteration, switch) in TASKS.items():
        original = file(f"payload/tasks/{name}/{basename}.pristine.py", "original")
        versions: list[dict[str, Any]] = [
            dict(
                id="original",
                files=[original],
                status="baseline",
                notes="Redacted snapshot of the historical original baseline.",
            )
        ]
        parent = "original"
        campaign = ROOT / "payload/campaigns" / name
        for path in sorted(campaign.glob("best/iter_*/publication.json")):
            publication = json.loads(path.read_text())
            commit, iteration = publication["commit_hash"], publication["iteration"]
            vid = f"kf-iter{iteration}-{commit[:8]}"
            versions.append(
                dict(
                    id=vid,
                    parent=parent,
                    files=[file(f"payload/campaigns/{name}/sources/{commit}.py")],
                    status="integrated"
                    if iteration == integrated_iteration and name != "sparse_mla"
                    else "candidate",
                    provenance=dict(
                        tool="KernelForge",
                        campaign=publication["campaign_id"],
                        commit=commit,
                        iteration=iteration,
                    ),
                    notes=publication["plan"],
                )
            )
            parent = vid
            if iteration == integrated_iteration:
                selected[name] = vid
        if name == "sparse_mla":
            versions.append(
                dict(
                    id="sglang-v0522-overlay",
                    parent=selected[name],
                    files=[file("payload/overlay/base/infera_kernels/kf_sparse_mla.py")],
                    status="integrated",
                    notes="Import-path compatibility adaptation; different source bytes.",
                    provenance=dict(derived_from=selected[name], tool="manual integration"),
                )
            )
            selected[name] = "sglang-v0522-overlay"
        for path in sorted(campaign.glob("rejected/iter_*/meta.json")):
            metadata = json.loads(path.read_text())
            source = path.parent / f"{basename}.py"
            versions.append(
                dict(
                    id=f"kf-iter{metadata['iteration']}-rejected",
                    parent=selected[name],
                    files=files([source]),
                    status="rejected",
                    provenance=dict(tool="KernelForge"),
                    notes="Correctness passed; reverted because performance did not improve the retained best.",
                )
            )
        limitations = [
            "New PyTorch reference has CPU semantic controls only; GPU differential validation is outstanding."
        ]
        if name == "dcp_indexer":
            limitations.append(
                "Synthetic correctness does not transfer to real inputs: archived real-input replay has mismatches and slow cases."
            )
        if name == "fused_indexer":
            limitations.extend(
                [
                    "Production byte-equality oracle remains authoritative; naive FP32 reductions may differ.",
                    "Latest KF iter10 is not the integrated iter4; no e2e improvement established.",
                ]
            )
        fixtures = [
            {k: item[k] for k in ("uri", "sha256", "size")}
            for item in external
            if f"/tasks/{name}/" in item["destination"]
        ]
        document = dict(
            kind="kernel",
            name=name,
            description=description,
            original=[original],
            reference=[
                file("references/naive.py"),
                file(f"payload/tasks/{name}/driver.py", "historical oracle"),
            ],
            reference_status="incomplete",
            limitations=limitations,
            documentation=files(
                [
                    f"payload/tasks/{name}/program.md",
                    "references/README.md",
                    "THIRD_PARTY_NOTICES.md",
                ]
            ),
            contract=dict(
                specification=file(f"payload/tasks/{name}/program.md"),
                naive_entrypoint="sparse_mla"
                if name == "dcp_sparse"
                else ("dsa_indexer" if name == "indexer" else name),
                fixtures=fixtures,
                hardware="gfx950",
                model="GLM-5.2-MXFP4",
            ),
            integration=dict(
                method="PYTHONPATH sitecustomize overlay with import hooks and per-kernel gates",
                version=selected[name],
                files=files(
                    [
                        "payload/overlay/base/sitecustomize.py",
                        "payload/overlay/base/infera_kernels/__init__.py",
                    ]
                ),
                activation=[
                    f"PYTHONPATH=<catalog>/payload/overlay/base:$PYTHONPATH INFERA_KERNEL_{switch}=1 <server command>"
                ],
                rollback=[f"Unset INFERA_KERNEL_{switch} and restart the server processes."],
                compatibility=dict(
                    sglang="0.5.21/0.5.22 paths as archived",
                    hardware="MI355X/gfx950",
                    guards="See exact adapter source for mode, shape and dtype restrictions.",
                    old_fused_module_adapter=file(f"{PD_FINAL}/kfov_fused/__init__.py"),
                ),
            ),
            versions=versions,
            campaign_history=files(sorted(p for p in campaign.rglob("*") if p.is_file())),
        )
        kernels[name] = document
        write(f"kernels/{name}/kernel.json", document)
    return kernels, selected


def sources(kernels, selected):
    result = []
    for name, version in selected.items():
        result.extend(next(v for v in kernels[name]["versions"] if v["id"] == version)["files"])
    return result


def base_run(name, kind, selected, kernels, env) -> dict[str, Any]:
    return dict(
        kind="run",
        name=name,
        description="Imported historical observation; no new GPU execution.",
        run_type=kind,
        timestamp="",
        kernels=selected,
        sources=sources(kernels, selected),
        baseline=[f for name in selected for f in kernels[name]["original"]],
        environment=env,
        baseline_environment=env,
        integration=[],
        workload={},
        configuration={},
        commands=[],
        correctness=dict(
            status="not_tested",
            scope="full-model" if kind == "e2e" else "operator",
            summary="No full-model correctness result accompanies this performance observation.",
            tolerances={},
        ),
        metrics=[],
        comparison=dict(controlled=False, confounders=[]),
        activation="not_applicable",
        evidence=[],
        reproducibility=dict(status="incomplete", missing=GAPS[:]),
        limitations=[],
        provenance=dict(imported=True, rerun=False),
    )


def scored_metric(benchmark):
    baseline = benchmark["baseline_case_times"]
    choices = []
    for repeat, measurement in enumerate(benchmark["measurements"]):
        cases = [
            dict(name=k, baseline=baseline[k], candidate=v)
            for k, v in measurement["case_times"].items()
            if k not in measurement.get("unscored_cases", []) and k in baseline
        ]
        choices.append(
            (sum(c["baseline"] / c["candidate"] for c in cases) / len(cases), repeat, cases)
        )
    score, repeat, cases = min(choices)
    if not math.isclose(score, benchmark["mean_case_speedup"], rel_tol=1e-8):
        raise ValueError("Retained repeat timings do not reproduce publication score")
    return dict(
        name="kernel_latency",
        unit="ms",
        direction="lower",
        aggregation="mean_case_ratio",
        cases=cases,
    ), repeat


def make_micro_runs(kernels):
    for name, (_, basename, _, _) in TASKS.items():
        campaign = ROOT / "payload/campaigns" / name
        for path in sorted(campaign.glob("best/iter_*/publication.json")):
            pub = json.loads(path.read_text())
            benchmark = json.loads((path.parent / "benchmark.json").read_text())
            vid = f"kf-iter{pub['iteration']}-{pub['commit_hash'][:8]}"
            run = base_run(
                f"kf-{name}-iter{pub['iteration']:03}",
                "microbench",
                {name: vid},
                kernels,
                "kf-workset",
            )
            run["description"] = pub["plan"]
            run["timestamp"] = pub["published_at"].replace(" ", "T")
            metric, repeat = scored_metric(benchmark)
            run["metrics"] = [metric]
            run["configuration"] = dict(
                cuda_graph=True,
                selected_repeat=repeat,
                aggregation_policy="minimum of repeated scored-case mean ratios",
                unscored_cases=benchmark["unscored_cases"],
                harness=file(f"payload/tasks/{name}/graph_harness.py"),
            )
            run["workload"] = dict(
                input_identity=dict(
                    driver=file(f"payload/tasks/{name}/driver.py"),
                    fixtures=kernels[name]["contract"]["fixtures"],
                ),
                definition=file(f"payload/tasks/{name}/program.md"),
                input_kind="synthetic activations/weights with captured workload metadata where supplied",
            )
            run["commands"] = [
                f"Stage version as {basename}.py beside the preserved driver and fixtures.",
                "python driver.py",
                "python driver.py --bench-mode",
            ]
            run["correctness"] = dict(
                status="passed" if pub["correctness_passed"] else "failed",
                scope="campaign synthetic operator cases",
                summary="Historical driver gate; not the newly authored catalog reference.",
                tolerances=dict(
                    observed_min_snr_db=pub["snr_db"],
                    gate=file(f"payload/tasks/{name}/campaign.env"),
                ),
            )
            run["comparison"]["controlled"] = True
            run["evidence"] = files(
                [path, path.parent / "benchmark.json", path.parent / "validation.txt"]
            )
            run["provenance"].update(
                campaign=pub["campaign_id"],
                commit=pub["commit_hash"],
                reported_mean_case_speedup=pub["mean_case_speedup"],
            )
            write(f"runs/{run['name']}.json", run)
        for path in sorted(campaign.glob("rejected/iter_*/meta.json")):
            meta = json.loads(path.read_text())
            vid = f"kf-iter{meta['iteration']}-rejected"
            run = base_run(
                f"kf-{name}-iter{meta['iteration']:03}-rejected",
                "microbench",
                {name: vid},
                kernels,
                "kf-workset",
            )
            run.update(
                timestamp=meta["timestamp"].replace(" ", "T"),
                description="Correct candidate reverted because it did not beat the retained best.",
                evidence=files(sorted(p for p in path.parent.iterdir() if p.is_file())),
                configuration=dict(decision=meta["decision"]),
                commands=["python driver.py", "python driver.py --bench-mode"],
                workload=dict(input_identity=file(f"payload/tasks/{name}/driver.py")),
                correctness=dict(
                    status="passed",
                    scope="campaign synthetic operator cases",
                    summary="Archived validation_passed=true; rejected for performance.",
                    tolerances={},
                ),
            )
            run["metrics"] = [
                dict(
                    name="reported_kernel_speedup",
                    unit="ratio",
                    direction="lower",
                    aggregation="reported",
                    reported_ratio=meta["mean_case_speedup"],
                )
            ]
            write(f"runs/{run['name']}.json", run)


def make_real_replay(kernels, selected):
    path = f"{DCP_PACK}/results/dcp_indexer_realdata.kdf1.txt"
    pattern = (
        r"(call\.\d+\.pt).*?prod ([\d.]+) ms kf ([\d.]+) ms.*?exact prod (\d+)/(\d+) kf (\d+)/(\d+)"
    )
    matches = re.findall(pattern, (ROOT / path).read_text())
    assert len(matches) == 24
    run = base_run(
        "dcp-indexer-real-input-replay",
        "microbench",
        {"dcp_indexer": selected["dcp_indexer"]},
        kernels,
        "historical-stack-a",
    )
    run.update(
        description="Real captured inputs include top-k mismatches and per-call regressions.",
        workload=dict(
            input_identity="kdf1/dcpdump capture (external, checksum unavailable)",
            input_kind="real capture",
        ),
        configuration=dict(calls=len(matches)),
        commands=["python dcp_indexer_realdata.registry.py"],
        correctness=dict(
            status="failed",
            scope="real-input top-k set equality",
            summary="At least one row differs in 11 of 24 captured calls; process exit 0 is not a correctness verdict.",
            tolerances=dict(require_identical_sets=True),
        ),
        evidence=files([path]),
        observed_metrics=[dict(call=m[0], exact_rows=int(m[5]), rows=int(m[6])) for m in matches],
    )
    failures = sum(int(m[5]) != int(m[6]) for m in matches)
    run["correctness"]["summary"] = (
        f"Top-k mismatches in {failures} of {len(matches)} captured calls; process exit 0 is not a correctness verdict."
    )
    run["metrics"] = [
        dict(
            name="latency",
            unit="ms",
            direction="lower",
            aggregation="mean_case_ratio",
            cases=[dict(name=m[0], baseline=float(m[1]), candidate=float(m[2])) for m in matches],
        )
    ]
    run["reproducibility"]["missing"].append("Real capture digest and durable retrieval URI")
    write(f"runs/{run['name']}.json", run)


def make_target_stack_micro(kernels, selected):
    log_path = f"{KF_PACK}/results/kbench.v0522.compute-node-06.log"
    content = (ROOT / log_path).read_text()
    for name, prefix in [
        ("sparse_mla", "sparse_mla"),
        ("indexer", "indexer"),
        ("moe_mxfp4", "moe"),
    ]:
        matches = re.findall(
            rf"^{prefix} (\S+): prod ([\d.]+) ms  KF ([\d.]+) ms.*?\| (.*)$", content, re.M
        )
        assert len(matches) == 4
        run = base_run(
            f"target-v0522-{name}",
            "microbench",
            {name: selected[name]},
            kernels,
            "historical-stack-a",
        )
        run.update(
            description="Archived in-process production versus overlay graph benchmark.",
            workload=dict(
                input_identity=file(f"payload/tasks/{name}/driver.py"), input_kind="synthetic"
            ),
            configuration=dict(cuda_graph=True, candidate="overlay", baseline="FORCE_ORIG"),
            commands=["python kbench.registry.py"],
            evidence=files([log_path]),
            comparison=dict(controlled=True, confounders=[]),
            correctness=dict(
                status="passed",
                scope="operator synthetic cases in target image",
                summary="; ".join(m[0] + ": " + m[3] for m in matches),
                tolerances={},
            ),
        )
        run["metrics"] = [
            dict(
                name="kernel_latency",
                unit="ms",
                direction="lower",
                aggregation="mean_case_ratio",
                cases=[
                    dict(name=m[0], baseline=float(m[1]), candidate=float(m[2])) for m in matches
                ],
            )
        ]
        run["limitations"] = [
            "Rounded printed timings are retained; ratio may differ slightly from printed speedup."
        ]
        write(f"runs/{run['name']}.json", run)
    for name, log_path, marker in [
        ("dcp_sparse", f"{DCP_PACK}/results/dcpk.v0522.compute-node-05.log", "dcp_sparse "),
        ("dcp_indexer", f"{DCP_PACK}/results/dcpk.v0522.compute-node-05.log", "dcp_indexer "),
        ("fused_indexer", f"{DCP_PACK}/results/fused_indexer_kernel.v0522.fik.log", ""),
    ]:
        content = (ROOT / log_path).read_text()
        baseline_section = content.split(f"=== {marker}prist\n", 1)[1].split("===", 1)[0]
        candidate_section = content.split(f"=== {marker}kf\n", 1)[1].split("===", 1)[0]

        def times(section):
            return {
                case: float(value)
                for case, value, unscored in re.findall(
                    r"^case_ms: (\S+) ([\d.]+)( unscored)?$", section, re.M
                )
                if not unscored
            }

        baseline, candidate = times(baseline_section), times(candidate_section)
        run = base_run(
            f"target-v0522-{name}",
            "microbench",
            {name: selected[name]},
            kernels,
            "historical-stack-a",
        )
        run.update(
            description="Archived target-image pristine versus candidate graph benchmark.",
            workload=dict(
                input_identity=file(f"payload/tasks/{name}/driver.py"), input_kind="synthetic"
            ),
            configuration=dict(cuda_graph=True),
            commands=["python driver.py", "python driver.py --bench-mode"],
            evidence=files([log_path]),
            comparison=dict(controlled=True, confounders=[]),
            correctness=dict(
                status="passed",
                scope="operator synthetic cases in target image",
                summary="Both retained sections report allclose=True; exact gate remains in the driver.",
                tolerances=dict(driver=file(f"payload/tasks/{name}/driver.py")),
            ),
        )
        assert "allclose: True" in candidate_section
        metrics: list[dict[str, Any]] = [
            dict(
                name="scored_case_latency",
                unit="ms",
                direction="lower",
                aggregation="mean_case_ratio",
                cases=[
                    dict(name=k, baseline=baseline[k], candidate=v) for k, v in candidate.items()
                ],
            )
        ]
        baseline_median = re.search(r"median_ms: ([\d.]+)", baseline_section)
        candidate_median = re.search(r"median_ms: ([\d.]+)", candidate_section)
        assert baseline_median is not None and candidate_median is not None
        metrics.append(
            dict(
                name="driver_median_latency",
                unit="ms",
                direction="lower",
                aggregation="ratio",
                baseline=float(baseline_median[1]),
                candidate=float(candidate_median[1]),
            )
        )
        run["metrics"] = metrics
        write(f"runs/{run['name']}.json", run)


def make_e2e_runs(kernels, selected):
    paths = sorted((ROOT / "payload/evidence").rglob("agentx_conc*.json"))
    observations, records = {}, {}
    for path in paths:
        if "runs" in path.parts:
            index = path.parts.index("runs")
            label = path.parts[index + 1] + "-" + path.parent.name.rsplit("-", 1)[-1]
        elif "rounds" in path.parts:
            index = path.parts.index("rounds")
            label = path.parts[index + 1] + "-" + path.parent.name.rsplit("-", 1)[-1]
        else:
            label = path.parent.name
        observations[label] = (path, json.loads(path.read_text()))
    for label, (path, data) in observations.items():
        if label.startswith(("kfa", "kda", "kdf", "ppda1", "ppda3")):
            continue  # Preserved baseline/capture evidence is referenced by candidate observations.
        chosen = {k: selected[k] for k in ("sparse_mla", "indexer", "moe_mxfp4")}
        if label.startswith("kdc"):
            chosen = {k: selected[k] for k in ("dcp_sparse", "dcp_indexer")}
        elif label.startswith("kde"):
            chosen = {"dcp_indexer": selected["dcp_indexer"]}
        elif label.startswith("kdg"):
            chosen = {"dcp_sparse": selected["dcp_sparse"]}
        if label.startswith(("kfc", "llr22-", "llr42-")):
            chosen["fused_indexer"] = selected["fused_indexer"]
        env = (
            "historical-stack-a"
            if label.startswith("k") or label.startswith("ppda")
            else "historical-stack-b"
        )
        if label.startswith("ppda"):
            env = "historical-stack-c"
        run = base_run(f"e2e-{label}", "e2e", chosen, kernels, env)
        pack_root = ROOT / "payload/evidence" / path.relative_to(ROOT).parts[2]
        related = [p for p in path.parent.iterdir() if p.is_file()]
        related += [pack_root / "README.md"]
        for extra in ["environment.md", "REPRODUCE.md"]:
            if (pack_root / extra).exists():
                related.append(pack_root / extra)
        run["evidence"] = files(related)
        run["description"] = "Historical serving measurement of the configured kernel bundle."
        profile = path.parent / "aiperf_artifacts/profile_export_aiperf.json"
        if profile.exists():
            run["timestamp"] = json.loads(profile.read_text()).get("start_time", "")
            run["evidence"].append(file(profile.relative_to(ROOT)))
        run["workload"] = dict(
            input_identity=data.get("dataset", {}),
            model_path=data.get("model"),
            model_identity=None,
            concurrency=data["conc"],
            input_kind="AgentX serving trace, simulated speculative acceptance",
        )
        run["configuration"] = {
            k: data.get(k)
            for k in (
                "image",
                "prefill_tp",
                "prefill_pp",
                "decode_tp",
                "decode_pp",
                "decode_dcp_size",
                "decode_ep",
            )
        }
        run["commands"] = [
            "Historical invocation is described by retained REPRODUCE.md and runtime.env; exact shell command not captured in this record."
        ]
        adapter = "payload/overlay/base"
        if label.startswith(("kfb1", "kfb2", "ppda2")):
            adapter = "payload/overlay/dpa-20261023"
        elif label.startswith("llr"):
            adapter = "payload/overlay/pd-20261024"
        run["integration"] = files(
            [f"{adapter}/sitecustomize.py", f"{adapter}/infera_kernels/__init__.py"]
        )
        if label.startswith("llr42-"):
            run["integration"][-1] = file(f"{PD_FINAL}/kfov_fused/__init__.py")
        run["activation"] = (
            "confirmed"
            if label.startswith(("kfb1", "kfb2", "kfc", "kdc", "kdg", "ppda2", "llr42-"))
            else "not_confirmed"
        )
        run["reproducibility"]["missing"] += [
            "Model checkpoint content identity",
            "Exact baseline target-image source digests",
            "Exact full invocation",
            "Per-run adapter identity for later pd configurations",
        ]
        request = data["request_metrics"]
        tpot = request.get("latency", {}).get("tpot", {}).get("mean")
        run["observed_metrics"] = dict(
            tpot_mean_ms=tpot * 1000 if tpot is not None else None,
            score_tokens_s_chip=request.get("throughput", {})
            .get("per_gpu", {})
            .get("total_tput_tps"),
            qps=request.get("qps", {}).get("mean"),
            request_accounting=data.get("request_accounting", {}),
        )
        run["limitations"] = [
            "The measurement concerns a bundle; per-kernel e2e attribution was not measured.",
            "Simulated acceptance and live kernel CHECK do not establish full-model correctness.",
        ]
        if tpot is None:
            run["limitations"].append("Archived result has no completed latency samples.")
        if label.startswith(("kdb", "llr22-")):
            run["activation"] = "not_confirmed"
            run["limitations"].append(
                "Requested optimization did not reach the intended execution path; reported no-op."
            )
        if label.startswith("llr42-"):
            run["limitations"].append(
                "Single run; underlying overlap/custom-AR stack later reported to garble under real acceptance."
            )
        if label.startswith("kdg"):
            run["comparison"]["confounders"].append(
                "Sparse-only run and baseline used different nodes."
            )
        paired = None
        for prefix, base in [
            ("kfb1", "kfa2"),
            ("kfb2", "kfa1"),
            ("kdc1", "kda1"),
            ("kdg1", "kda1"),
            ("kfc1", "kfb3"),
        ]:
            if label.startswith(prefix):
                paired = base + "-" + label.rsplit("-", 1)[-1]
        if paired in observations:
            baseline_path, baseline_data = observations[paired]
            run["baseline"] += files([baseline_path])
            run["metrics"] = [
                dict(
                    name="tpot_mean",
                    unit="ms",
                    direction="lower",
                    aggregation="ratio",
                    baseline=baseline_data["request_metrics"]["latency"]["tpot"]["mean"] * 1000,
                    candidate=run["observed_metrics"]["tpot_mean_ms"],
                )
            ]
            run["comparison"]["controlled"] = label.startswith(("kfb1", "kfb2", "kdc1"))
            if label.startswith("kfc"):
                run["comparison"]["confounders"].append(
                    "Fused and three-kernel measurements used different nodes."
                )
        else:
            run["comparison"]["confounders"].append(
                "No isolated paired baseline assigned; raw values are retained."
            )
        write(f"runs/{run['name']}.json", run)
        records[label] = run
    if "llr7-c80" in records:
        run = records["llr7-c80"]
        run["metrics"] = [
            dict(
                name="reported_bundle_score_gain",
                unit="ratio",
                direction="higher",
                aggregation="reported",
                reported_ratio=41679 / 37743,
            )
        ]
        run["baseline"].append(file(f"{PD_PACK}/results/llr/summary.txt"))
        run["comparison"]["confounders"] = [
            "Custom all-reduce/top-k settings changed; mean input length grew 3.6%.",
            "Baseline and candidate score values are rounded in the archived summary.",
        ]
        write(f"runs/{run['name']}.json", run)
    if "llr42-c104" in records:
        run = records["llr42-c104"]
        run["metrics"] = [
            dict(
                name="reported_bundle_score_ratio",
                unit="ratio",
                direction="higher",
                aggregation="reported",
                reported_ratio=47674 / 48408,
            )
        ]
        run["baseline"].append(file(f"{PD_FINAL}/README.md"))
        run["comparison"]["confounders"] = [
            "Single candidate run versus two baseline runs; rounded summary values."
        ]
        write(f"runs/{run['name']}.json", run)
    interrupted = base_run(
        "e2e-kde1-A1-interrupted",
        "e2e",
        {"dcp_indexer": selected["dcp_indexer"]},
        kernels,
        "historical-stack-a",
    )
    interrupted.update(
        description="DCP indexer-only serving run stopped before aiperf exported complete JSON.",
        workload=dict(input_identity="AgentX serving trace", concurrency=56, model_identity=None),
        configuration=dict(topology="DCP4", enabled="DCP indexer only"),
        commands=["Historical serving harness; exact invocation not captured in this record."],
        evidence=files([f"{DCP_PACK}/results/summary.tsv", f"{DCP_PACK}/notes.md"]),
        integration=files(
            [
                "payload/overlay/base/sitecustomize.py",
                "payload/overlay/base/infera_kernels/__init__.py",
            ]
        ),
        activation="confirmed",
        observed_metrics=dict(tpot_mean_ms=21.51, score_tokens_s_chip=None),
        limitations=[
            "Interrupted run: no complete throughput export or full-model correctness result.",
            "Known real-input DCP indexer mismatch is recorded separately.",
        ],
    )
    write(f"runs/{interrupted['name']}.json", interrupted)
    return records


def main():
    verify_payload()
    make_environments()
    kernels, selected = make_kernels()
    make_micro_runs(kernels)
    make_real_replay(kernels, selected)
    make_target_stack_micro(kernels, selected)
    make_e2e_runs(kernels, selected)
    report = Registry(ROOT).validate()
    print(json.dumps({k: v for k, v in report.items() if k != "warnings"}, indent=2))
    print(f"Historical completeness warnings: {len(report['warnings'])}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
