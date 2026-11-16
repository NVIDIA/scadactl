# Changelog

All notable changes to `scadactl` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Apache-2.0 project licensing and SPDX source headers.
- Standalone Python packaging and Debian system packaging.
- `manage-scada-domains` as a project-owned runtime component.
- Security, support, governance, contribution, and release documentation.
- GitHub issue and pull-request templates.

### Changed

- Kernel module loading and unloading are explicit UNVMe runtime capabilities.
- NVMe aliases are derived from the enumerated controller device name instead
  of PCI ordering.
- Controllers exposing multiple namespaces are rejected because the persisted
  serial and PCI identity cannot distinguish their namespaces safely.
