# Usage statistics

Infera supports optional outbound deployment usage reporting. Its reporting
policy is **enabled unless opted out**, but this version ships **without a
shared collector URL**. No data is collected or sent until a deployment sets
`INFERA_USAGE_STATS_SERVER` to its HTTPS collector. A project-owned collector
must be provisioned before default-on reporting works out of the box.

## Scope and lifecycle

The Python server (`python -m infera.server`) and the vLLM, SGLang and ATOM
worker launchers participate. The Rust router, standalone kvd, simulation,
benchmark and preflight commands do not report Infera usage.

Workers start reporting after successful registration and readiness setup.
The Python server starts reporting as it enters its serving loop; a session
start is not evidence that the HTTP socket successfully bound. Each session
sends a `session_start` event, then a `heartbeat` every 600 seconds while
serving. Reporting stops before worker draining or server cleanup. There is
no exit report, so a missing heartbeat does not establish a crash.

Delivery runs in a background task with a two-second total timeout per
attempt. Failed events are dropped, redirects are not followed, and payloads
are never written to disk. No request handlers perform reporting work.

## Disable reporting

Any one of these disables Infera reporting, even with a collector configured:

```bash
export INFERA_NO_USAGE_STATS=1
# Or a cross-project preference:
export DO_NOT_TRACK=1
# Or:
export TELEMETRY_DISABLED=true
```

These controls accept `1`, `true`, `yes`, or `on` (case insensitive). Set them
on every server and worker process, including container environments.
Alternatively, create the preference file before launch:

```bash
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/infera"
touch "${XDG_CONFIG_HOME:-$HOME/.config}/infera/do_not_track"
```

The file and environment are checked before each event. Adding the file stops
future reports; it cannot recall an event already in flight. Removing an
opt-out requires a restart to resume reporting. Reporting is also disabled
under CI and pytest (`CI`, `GITHUB_ACTIONS`, `GITLAB_CI`, `JENKINS_URL`, or
`PYTEST_CURRENT_TEST` set to a nonempty value other than `0` or `false`).

Infera's setting controls Infera reporting only. Engines may have independent
usage collection. For vLLM, use `VLLM_NO_USAGE_STATS=1` or its documented
`DO_NOT_TRACK` support as well. See the
[vLLM usage documentation](https://docs.vllm.ai/en/stable/usage/usage_stats/).

## Collected fields (schema version 1)

Every JSON event has exactly these fields:

| Field | Value |
| --- | --- |
| `schema_version` | Integer `1` |
| `event` | `session_start` or `heartbeat` |
| `session_id` | Random UUID hex string, new for every serving session |
| `sequence` | Zero-based event attempt number |
| `elapsed_seconds` | Whole seconds since reporting started |
| `infera_version` | Numeric package release (`major.minor.patch`) or `unknown` |
| `python_version` | Python version |
| `os` | `Linux`, `Windows`, `Darwin`, or `unknown` |
| `cpu_architecture` | `x86_64`, `aarch64`, `AMD64`, `arm64`, or `unknown` |
| `component` | `server`, `worker`, or `unknown` |
| `engine` | `vllm`, `sglang`, `atom`, `none`, or `unknown` |
| `disagg_mode` | `mixed`, `prefill`, `decode`, `none`, or `unknown` |
| `request_transport` | `http`, `nats`, or `unknown` |

No prompts, outputs, tokens, model names or paths, hostnames, credentials,
command lines, arbitrary configuration, exception text, persistent machine
identifiers, or GPU identifiers are included. The collector sees the source
IP at the network layer; a session UUID is not a guarantee of anonymity.
Collection and retention of HTTP access logs are the collector operator's
responsibility.

## Collector configuration

Set `INFERA_USAGE_STATS_SERVER` to the complete HTTPS ingestion URL. An empty
or invalid URL disables reporting. Credentials in the URL and URL fragments
are rejected. TLS certificate validation remains enabled. Standard HTTP proxy
environment settings apply.

The collector should accept JSON `POST` requests without requiring redirects
and return a success status. This client does not provision a collector,
storage, access controls, dashboards, or a retention policy. Configure these
for your deployment before enabling ingestion. No third-party project endpoint
is used as an Infera default.

At startup, Infera logs whether reporting has an active collector and how to
opt out. Its payload schema is deliberately small and categorical; new fields
should receive an explicit privacy review and corresponding schema tests.

This design follows the configurable endpoint and opt-out patterns in
[vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/usage/usage_lib.py)
and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/usage/schemas/README.md).
