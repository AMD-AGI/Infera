Ready for review with signed test-only follow-up `5cee57e0`; the production fix is unchanged.

Added seven CPU-built ELF fixture cases that run the production artifact checks with real `nm`, `ldd` and `strings`, covering the valid artifact and six missing-capability diagnostics. With the original four regressions, **11 pass**; the unpatched artifact-check block produces **9 failures and 2 passing missing-symbol controls**. Shell syntax and applicable pre-commit checks pass; CI is running on this head.

The body now separates review readiness from broader integration limits. These fixtures are not a Mooncake/HSA runtime, and the complete image has not been rebuilt. The focused shell check is supported by direct before/after and actual binary-inspection coverage; no GPU or serving result is claimed.
