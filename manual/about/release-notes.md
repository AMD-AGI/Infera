# Release notes

## Infera {{ infera_version }} documentation

This documentation update consolidates installation, concepts, reference
material, and examples in one Infera site.

### Added

- An installation guide with ROCm version and installation method selectors.
- Prerequisites naming the GPU targets, engine requirements, and ROCm
  compatibility matrix.
- Version-specific ROCm installation links and a dedicated ROCm 10.2 SGLang
  container build example.

### Changed

- The site header, installation guide, and compatibility matrix identify
  Infera {{ infera_version }}.
- Navigation uses Install, Conceptual, Reference, Examples, and About.
  Recipes appear under Examples; simulation documentation appears under
  Conceptual.
- The home page links directly to installation. Existing page paths are retained
  so links to the home page and individual guides continue to work.
- The compatibility matrix identifies ROCm 10.2 as a validated configuration
  and retains the exact pre-release image provenance and scenario limits.

### Known issues

The [compatibility matrix](../features/compatibility_matrix.md#rocm-102-validation)
records an SGLang 0.5.21 disaggregated-decode defect after long requests on the
ROCm 10.2 image. Use its aggregated configuration. Other engine, model, GPU,
and cross-node combinations remain outside that recorded validation scope.

These are documentation release notes. They do not announce new engine fixes
or expand the validation scope beyond the compatibility matrix. For software
release artifacts, see [Infera v0.3.3 on GitHub](https://github.com/AMD-AGI/Infera/releases/tag/v0.3.3).
