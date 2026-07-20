# Minimal Sphinx Makefile for the Optimus manual.
# Usage: make html  |  make clean  |  make linkcheck  |  make serve
SPHINXOPTS    ?= -W --keep-going -n
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean linkcheck serve
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "\nBuilt: $(BUILDDIR)/html/index.html"

clean:
	@rm -rf "$(BUILDDIR)"

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Live-reload preview (pip install sphinx-autobuild).
serve:
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --port 8800
