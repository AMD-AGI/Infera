# My kernel catalog

Replace the example with your own artifacts. Edit the three JSON document types
and commit with Git. Keep readable version labels, original source notices,
naive references, exact integration code and raw test results.

The supplied run documents are **unmeasured templates**, not test evidence.
Capture an environment in the environment under test, fill the commands and
results, then use a new run name for each measurement. For an incomplete historical
import, list what is missing instead of inventing it.

Use `kernel-registry file . payloads/candidate.py` after adding a file,
`kernel-registry validate .` to check links/hashes and see information gaps,
and `kernel-registry history .` to inspect observations. Large inputs can use
external URI/checksum/size descriptors; store them durably in your own storage.
