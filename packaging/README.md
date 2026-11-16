# System package layout

The Debian package in `debian/` installs these project-owned files:

| Repository path | Installation path |
| --- | --- |
| `packaging/scadactl` | `/usr/bin/scadactl` |
| `scadactl/` | `/usr/lib/scada-tools/scadactl/` |
| `scripts/manage-scada-domains` | `/usr/sbin/manage-scada-domains` |

Build it on a Debian-compatible system with:

```sh
dpkg-buildpackage -us -uc -b
```

The package runs the hardware-free Python test suite during the build and
provides an autopkgtest smoke test for the installed commands.

`scada-load-modules` and `scada-unload-modules` are kernel-backend capabilities
installed with the UNVMe runtime. `scadactl` discovers them at `/usr/sbin` or
uses explicit absolute paths supplied through `SCADA_LOAD_SCRIPT` and
`SCADA_UNLOAD_SCRIPT`.
