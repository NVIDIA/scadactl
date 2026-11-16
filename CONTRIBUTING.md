# Contributing to scadactl

`scadactl` welcomes focused bug reports, documentation corrections, design
proposals, and code contributions. By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Before contributing

Do not report security vulnerabilities through a public issue or pull request.
Follow [SECURITY.md](SECURITY.md) and use NVIDIA PSIRT's private reporting
channels.

Search existing issues before opening a new one. Discuss substantial CLI,
configuration-schema, security-boundary, device-lifecycle, or packaging changes
with maintainers before implementation.

Submissions are evaluated for technical fit, safety, maintainability,
compatibility, and project priorities. Submission does not guarantee acceptance.

## Contribution terms

`scadactl` is licensed under Apache-2.0. Unless explicitly stated otherwise,
contributions are submitted under the same license. A Developer Certificate of
Origin (DCO) sign-off is the required legal attestation for a contribution.

## Signing Your Work

Every commit must include a sign-off. Use `git commit -s` when committing:

```sh
git commit -s -m "Describe the change"
```

This adds a `Signed-off-by: Name <email>` trailer and certifies the Developer
[Certificate of Origin 1.1](https://developercertificate.org/):

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Commits without sign-off are not accepted.

The DCO document above retains its own copyright and verbatim-copying terms;
those terms apply to the DCO text, not to the application's Apache-2.0 license.

Preserve third-party notices and identify third-party material, its source, and
its license in the pull request. Do not submit material you do not have the right
to contribute.

## Development setup

Create an isolated environment and install the development dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
bash -n scripts/manage-scada-domains
shellcheck scripts/manage-scada-domains
python3 -m build
```

The Python tests do not require a GPU or root. Tests that mutate GPU, NVMe,
driver, MPS, or domain state require a dedicated, explicitly provisioned host.
Never run device-changing tests against a boot or data-bearing controller.

Build the Debian package when packaging files change:

```sh
dpkg-buildpackage -us -uc -b
```

## Pull requests

Keep changes focused and include:

- the problem and intended behavior;
- tests for behavior changes and documentation for changed contracts;
- security, compatibility, device-safety, and packaging impact;
- third-party provenance when applicable; and
- the exact commands and results used for verification.

Maintainer and CODEOWNERS approval is required. Reviewers may require additional
hardware, packaging, security, legal, licensing, or provenance evidence.
