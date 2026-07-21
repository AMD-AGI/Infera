###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Configuration file for the Infera user manual.
#
# Build:  cd manual && pip install -r requirements.txt && make html
# Output: manual/_build/html/index.html
#
# This is the user-facing manual (getting started, serving, feature
# one-pagers, reference). Keep code-internal design detail out of this tree;
# if a design point is useful to a user, inline the relevant part here.

author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) %Y Advanced Micro Devices, Inc. All rights reserved."
release = "0.1"
version = release
project = f"Infera {version}"

# Theme-related configs
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "ai-ecosystem",
    "link_main_doc": True,
    "repository_url": "https://github.com/AMD-AGI/Infera",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}
html_title = project

# Sphinx extension-related configs
extensions = [
    "rocm_docs",
    "sphinx.ext.graphviz",
]
external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "infera"
# Uncomment if you encounter rate limits when building locally
# external_projects_remote_repository = ""

# Render Graphviz to SVG (scales crisply, selectable text) rather than PNG.
graphviz_output_format = "svg"

# MyST niceties: ::: fenced directives, auto-linkify bare URLs, $math$, etc.
myst_enable_extensions = {
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
    "tasklist",
}
myst_heading_anchors = 3  # auto-generate anchors for h1..h3 so cross-links work

# Don't choke the build on a mistyped cross-reference; warn instead.
nitpicky = False
suppress_warnings = ["myst.header"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", ".venv", ".venv/**"]

# Publish the llms.txt index at the docs site root and let
# rocm-docs-core generate llms-full.txt after each build (the llms.txt standard,
# https://llmstxt.org/). See the rocm-docs-core guide:
# https://rocm.docs.amd.com/projects/rocm-docs-core/en/latest/user_guide/llms.html
rocm_docs_generate_llms = True
