# Round002 — initial model launch

Target: actual8 experts, TP4/DPA/EP4, healthy1P1D.
Result: FAIL before model loading. Both legs reject --json-model-override-args with JSONDecodeError extra data.
Evidence: launch-driver.log emitted {"index_share_for_mtp_iteration":false}}; standalone sourced-config output reproduces extra trailing brace. Vendored config.sh default parameter expansion adds brace to pre-set JSON. User reference assigns JSON override AFTER sourcing, whereas our initial config did it before.
Fix: reapply exact JSON after source in isolated config.yihou.expert8.sh. Validated with python -m json.tool. No SGLang patch changed. Own failed containers stopped; logs preserved. Next round004 retries with corrected JSON after clean baseline and GPU RDMA precheck.
