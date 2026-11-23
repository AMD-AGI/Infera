"""o11y_facts — the facts builder (design §8.0).

`python -m o11y_facts` serves queue kind `display`: it reads the sources (through `o11y_sources`, with its
own cursors), the ledger and the events, and writes `facts/{general,detail,kernelforge,usage}.json`; private
state in `state/facts/`. It never imports `o11y_log` and writes no events or ledger records.
"""
