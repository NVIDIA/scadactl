# Third-party notices

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Project copyright: Copyright 2025-2026 NVIDIA Corporation and affiliates.
All rights reserved.

NVIDIA-authored project code is licensed under Apache-2.0; see [LICENSE](LICENSE).
The third-party material identified below retains its own copyright and terms.
These notices do not relicense that material under Apache-2.0.

## Third-party material included in this distribution

### Developer Certificate of Origin 1.1

Location: [CONTRIBUTING.md](CONTRIBUTING.md), which reproduces the full DCO text.

Source: <https://developercertificate.org/>

The document's copyright and copying terms are reproduced verbatim:

```text
Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.
```

The DCO is a contribution attestation, not the license for scadactl application
code. The original DCO text has not been modified.

### Contributor Covenant 1.4

Location: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Author and attribution: Coraline Ada Ehmke and Contributor Covenant contributors.

Source: <https://www.contributor-covenant.org/version/1/4/code-of-conduct.html>

License: Creative Commons Attribution 4.0 International (CC-BY-4.0).

- License: <https://creativecommons.org/licenses/by/4.0/>
- Full legal terms: <https://creativecommons.org/licenses/by/4.0/legalcode>

The project's Code of Conduct is adapted from Contributor Covenant 1.4;
wording and enforcement details have been changed. The source attribution and
license reference are also retained in that document. No endorsement by the
original authors is implied. These terms apply to the adapted Code of Conduct,
not to the application's Apache-2.0 code.

## Separately installed software dependencies

The following dependencies are identified for transparency. Their source and
binaries are not bundled in this source distribution. Obtain their copyright
notices and complete license terms from the linked upstream sources and the
installed packages. This list is not a grant of rights to redistribute those
packages. See [DEPENDENCIES.md](DEPENDENCIES.md) for capability requirements.

### Runtime and host packages

| Component | Applicable license | Upstream license or copyright reference |
| --- | --- | --- |
| Python 3.10+ | PSF License and historical terms retained by Python | [Python license](https://docs.python.org/3.10/license.html) |
| PyYAML | MIT | [License](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| Optional `nvidia-ml-py` 12+ | BSD-3-Clause | [Project licensing](https://pypi.org/project/nvidia-ml-py/) |
| Bash 4+ | GPL-3.0-or-later | [Project licensing](https://www.gnu.org/software/bash/bash.html) |
| `pciutils` (declared Debian dependency) | GPL-2.0-or-later | [License](https://github.com/pciutils/pciutils/blob/master/COPYING) |
| `procps`: `pgrep`, `pkill` | GPL-2.0-or-later | [Source copyright and license](https://github.com/procps-ng/procps/blob/master/src/pgrep.c) |
| `psmisc`: `fuser` | GPL-2.0-or-later | [License](https://gitlab.com/psmisc/psmisc/-/blob/master/COPYING) |
| `util-linux`: `lsblk` | GPL-2.0-or-later | [Source copyright and license](https://github.com/util-linux/util-linux/blob/v2.39.3/misc-utils/lsblk.c) |
| GNU coreutils: `cat`, `date`, `id`, `mkdir`, `readlink`, `realpath`, `rm`, `sleep`, `sort` | GPL-3.0-or-later | [Source copyright and license](https://github.com/coreutils/coreutils/blob/master/src/readlink.c) |
| GNU `sed` | GPL-3.0-or-later | [Source copyright and license](https://github.com/mirror/sed/blob/master/sed/sed.c) |
| `awk` implementation: `mawk` or GNU `gawk` | mawk: GPL-2.0-only; gawk: GPL-3.0-or-later | [mawk copyright and license](https://github.com/ThomasDickey/mawk-snapshots/blob/master/main.c); [gawk licensing](https://www.gnu.org/software/gawk/manual/gawk.html) |
| Optional `nvme-cli` | GPL-2.0-only | [License](https://github.com/linux-nvme/nvme-cli/blob/master/LICENSE) |

### Build and test packages

| Component | Applicable license | Upstream license or copyright reference |
| --- | --- | --- |
| `setuptools` 61+ | MIT | [License](https://github.com/pypa/setuptools/blob/main/LICENSE) |
| `build` | MIT | [License](https://github.com/pypa/build/blob/main/LICENSE) |
| `pytest` | MIT | [License](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
| GNU Make | GPL-3.0-or-later | [Source copyright and license](https://github.com/mirror/make/blob/master/src/main.c) |
| `dpkg-dev`, including `dpkg-buildpackage` | GPL-2.0-or-later | [Copyright and licenses](https://github.com/guillemj/dpkg/blob/main/debian/copyright) |
| Optional ShellCheck | GPL-3.0-or-later | [License](https://github.com/koalaman/shellcheck/blob/master/LICENSE) |
| debhelper: `dh`, sequencing/build-system modules, preparation/automatic-build/file-installation/content-processing/archive-generation helpers listed in DEPENDENCIES.md | GPL-2.0-or-later | [Component copyright and licenses](https://sources.debian.org/src/debhelper/14.5/debian/copyright/) |
| debhelper: default-sequence `dh_installinit` | GPL-3.0-or-later | [Component copyright and licenses](https://sources.debian.org/src/debhelper/14.5/debian/copyright/) |
| debhelper: `autoscripts/*`, if generated during Debian packaging | Public domain, with an alternative permissive grant in the upstream copyright file | [Copying terms](https://sources.debian.org/src/debhelper/14.5/debian/copyright/) |

The release described here is source-only and does not include generated Debian
packages or maintainer scripts. Any later distribution bundling dependencies or
generated third-party material must carry the notices and license terms required
for that distribution.

Externally installed NVIDIA drivers, CUDA components, SCADA domain servers, and
storage backends are not included or relicensed by this source release.
