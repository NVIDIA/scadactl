# SCADA Orchestrator (`scadactl` CLI)

The `scadactl` command is the user-facing control plane for configuring, launching,
validating, and using SCADA on hosts with GPUs and NVMe storage. It keeps four
concepts separate: **inventory** (what exists), **config** (what the operator
wants, stable-identifier keyed), **registry** (what is running), and **client
selection** (how an application connects).

The package imports and the hardware-free commands run on a machine with **no
GPU and no root**. Hardware/driver/root-dependent paths are implemented against
the real interfaces but lazy-imported and guarded so they degrade gracefully.

## Layout

```text
.
  pyproject.toml                 PEP 621 project; console_script `scadactl`
  README.md                      this file
  scripts/
    manage-scada-domains         launch and stop scada-swap-domain processes
  scadactl/
    cli.py                       argparse dispatch + --help for every command
    models.py                    dataclasses: Inventory, Config, Registry
                                 (+ schema_version constants)
    inventory.py                 GPU (NVML) + drive (/sys, nvme, vfio-pci) enumeration
    identity.py                  alias gen, alias<->id resolution, UUID->index,
                                 UNVMe correlation (sysfs primary + fallback)
    config.py                    YAML load/validate/author (all validation rules)
    plan.py                      desired-vs-registry diff + command rendering
    apply.py                     reconcile: validate->backend setup->MPS->launch->publish
    registry.py                  atomic locked read/write + reconcile semantics
    policy.py                    host device allow/block lists (inventory annotation)
    client.py                    container ordinal mapping; run/env/domains/check-client
    diagnostics.py               status / doctor / logs
    backend/
      binding.py                 wraps scada-load-/scada-unload-modules
      mps.py                     calls nvidia-cuda-mps-control directly
      launcher.py                wraps manage-scada-domains (one call per domain)
  tests/                         pytest; hardware-free, no GPU/root needed
```

## Install

For a Python-only development or inspection environment, run from the repository
checkout root:

```shell
python3 -m pip install -e .            # installs the Python CLI and PyYAML
python3 -m pip install -e '.[gpu]'     # installs nvidia-ml-py (provides the pynvml module) for GPU-dependent commands
python3 -m pip install -e '.[dev]'     # also install build and pytest
```

`pynvml` is an **optional, lazy** import. Without it the package still imports
and all hardware-free commands work; GPU-dependent commands fail with an
actionable "install `scadactl[gpu]`" message.

The Python wheel installs the `scadactl` console command but not the privileged
domain manager. A complete host installation uses the Debian package, which
installs the CLI, Python modules, and `manage-scada-domains`:

```shell
dpkg-buildpackage -us -uc -b
sudo apt install ../scadactl_0.6.0-1_all.deb
```

Storage drivers, backend capabilities, and `scada-swap-domain` are installed
separately. See [DEPENDENCIES.md](DEPENDENCIES.md) for the complete runtime
boundary and [packaging/README.md](packaging/README.md) for installed paths.

## Commands

Configs are managed, name-addressable objects in a store at `/etc/scada/configs`
(`init`/`edit`/`import` save there; `list`/`show` read it). Every
config-consuming command takes either a bare **store NAME** or **`-f FILE`**
(kubectl-style; mutually exclusive).

| Command | Purpose |
| --- | --- |
| `scadactl inventory [--json]` | Show GPUs, drives, aliases, topology, binding state |
| `scadactl config init [--empty\|--gpus\|--drives] [-o FILE] [--force]` | Create a config (saves to the store) |
| `scadactl config list [--json]` | List stored configs (name, domains, created/modified, active) |
| `scadactl config show NAME\|-f FILE [--json]` | Show a config's details |
| `scadactl config edit NAME\|-f FILE` | Interactively edit a config |
| `scadactl config rm NAME` | Delete a stored config |
| `scadactl config import FILE [--force]` | Register an external config file into the store |
| `scadactl config compat <configs…> [-f FILE…] [--json]` | Can these configs run concurrently? |
| `scadactl config validate NAME\|-f FILE [--json]` | Validate (correctness/lint) |
| `scadactl config add-domain --name --gpu --drives [-f FILE]` | Add a domain (aliases in, stable ids out) |
| `scadactl policy block\|unblock [--gpus …] [--drives …] [--reason …]` | Mark devices off-limits / lift it (admin) |
| `scadactl policy allow\|unallow [--gpus …] [--drives …]` | Confine SCADA to an allow-list / lift it (admin) |
| `scadactl policy list [--json]` | Show the effective device policy per device |
| `scadactl policy clear [--all\|--block\|--allow]` | Clear policy lists (admin) |
| `scadactl apply NAME\|-f FILE [--dry-run [--commands] [--json]] [--force]` | Preview (`--dry-run`) or reconcile the host (`--force` overrides policy) |
| `scadactl stop --config NAME \| --all [--dry-run]` | Stop managed domains |
| `scadactl status [--details] [--json]` | Active domains, sockets, health, drives |
| `scadactl logs [--domains a,b] [--json]` | Per-domain log paths (newest run first) |
| `scadactl domains [--json]` | Domains usable from the current environment |
| `scadactl run --domain\|--domains -- CMD` | Run with process-local CUDA + SCADA env |
| `scadactl env --domain\|--domains [--json]` | Print exports / JSON (`domain_count`) |
| `scadactl check-client --domain NAME` | Container visibility diagnostic + docker fix |
| `scadactl doctor [--logs]` | Host + domain checks, registry vs live state |

Preview model (kubectl-style): **`validate`** answers "is the config correct?";
**`apply --dry-run`** is the single preview of what would change (add `--commands`
for the exact backend invocations, `--json` for machine output); **`apply`**
executes.

### Reconciliation (what `apply` does on re-run)

`apply` **reconciles the delta** toward a config's desired state instead of
blindly replaying setup. On re-run, each resource resolves to one of these outcomes:

- **Unchanged re-apply** (config matches the registry, all domains healthy) → **no-op**: exits success, changes nothing.
- **Edited active config** → **refused**: an active config is immutable in place (the registry, not the edited file, is the comparison basis). Stop the config first (`scadactl stop --config <name>`), then apply the new version.
- **Resource held by another active config** → **hard error**: `apply` never seizes a managed resource. Stop the owning config first.
- **Unmanaged orphan** (on host but not in registry) → **reclaimed**: torn down and rebuilt. `apply --dry-run` warns loudly before this happens.
- **Dead managed domain** (recorded in registry but process died) → **self-healed**: only the missing pieces are rebuilt.
- **Unvme drive held by live unregistered process** → **error**: cannot be freed (no device-targeted backend kill yet). Fix manually or try a different config. This kernel-only reclaim rule does not apply to Storage Lender domains.

Device policy (admin, host-wide): mark physical devices off-limits or confine
SCADA to an allow-list. `scadactl policy block --gpus gpu2 --reason "reserved"` adds
to the block list; `allow` builds a default-deny allow-list; `unblock`/`unallow`
lift them; `policy list` shows the effective decision per device. Blocked devices
are flagged in `inventory` (the drive `status` column, where a block dominates
mount-availability), in the config wizard, and by `validate`; `apply` refuses them
unless you pass `--force`. Policy lives at `/etc/scada/policy.yaml` (override
`SCADA_POLICY_FILE`) — world-readable, root-writable, so reading is unprivileged
but changing it needs root. An empty or absent policy changes nothing.

Drives carry capacity as both a humanized `size` (e.g. `3.2TB`) and a raw
`size_bytes`; the `inventory` Drives table shows the humanized form in a `size`
column between `model` and `lba`, and the config wizard's drive-selection table
shows it too. Capacity is sourced from `nvme list -o json`'s `PhysicalSize` for
drives still on the `nvme` driver, and from the unvme driver's `size_bytes` sysfs
attribute (`/sys/class/unvme/unvme<N>/size_bytes`) for drives already bound to
`unvme`; either may be absent/`unknown` on older drivers, in which case both
fields are `None` and the table cell is blank (never an error).

The runtime directory defaults to `/var/run/scada` (override `SCADA_RUNTIME_DIR`),
and the config store to `/etc/scada/configs` (override `SCADA_CONFIG_DIR`; both
used by the tests). The domain manager defaults to the project-owned
`/usr/sbin/manage-scada-domains`. Kernel-backed storage additionally requires
the UNVMe runtime's `/usr/sbin/scada-load-modules` and
`/usr/sbin/scada-unload-modules` capabilities. Override these paths with
`SCADA_MANAGE_SCRIPT`, `SCADA_LOAD_SCRIPT`, and `SCADA_UNLOAD_SCRIPT`; overrides
must be absolute. MPS is driven by calling `nvidia-cuda-mps-control` directly;
override the binary with `SCADA_MPS_CONTROL_BIN` and the pipe/log directories
with the standard `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY`. Server
logs are timestamped per launch:
`/var/log/scada/<config>/<domain>/scada-swap-domain.gpu<N>.<stamp>.log`.

## Root servers and non-root clients

The domain server runs as root. To allow non-root clients, create a Unix group
and add the intended users. Clients also need read/write access to
`/dev/udmabuf`; on Ubuntu 24.04 this normally means membership in `kvm`.
Check the device's actual group with `ls -l /dev/udmabuf`.

```bash
sudo addgroup --system scada-users
sudo usermod -aG scada-users,kvm <user>
```

Start a new login session and verify membership with `id`. In `scadactl config
init`, select `scada-users` for each domain's **Client socket group**. The same
setting is available when editing a domain, in YAML as `socket_group`, or with
`config add-domain --socket-group scada-users`. Leaving the group blank retains
root-only socket access. When editing, blank keeps the current group; enter `-`
to restore root-only access. The socket is owned by `root:<group>` with mode `0660`.
Changing a running domain's group requires stopping its config and applying it
again; editing a config file alone does not change the running server.

When MPS is enabled, a root server and non-root GPU clients require **multiuser
MPS**. The wizard defaults to multiuser mode for new configs and when enabling
MPS; editing an existing enabled config preserves its current mode as the
prompt default. Accept the default or set:

```yaml
mps:
  enabled: true
  multiuser: true
domains:
  - name: gpu0_domain
    socket_group: scada-users
    # Keep the GPU identity, drives, and other fields generated by config init.
```

This is a fragment to merge into the generated config, not a complete config.
Then preview and apply it as usual:

```bash
sudo scadactl apply -f config.yaml --dry-run --commands
sudo scadactl apply -f config.yaml
# In the authorized user's new login session:
scadactl run --domain gpu0_domain -- /usr/bin/vectoradd
```

Multiuser MPS starts the root control daemon with
`nvidia-cuda-mps-control -d -multiuser-server`. It removes user isolation within
that MPS instance, so use it only with trusted users; a fatal client GPU fault
can affect other users' clients. The SCADA socket group still controls access
to the SCADA domain. See NVIDIA's [MPS architecture documentation](https://docs.nvidia.com/deploy/mps/architecture.html).

Existing configs default to single-user MPS when `multiuser` is omitted. In that
mode, a non-root GPU client can wait indefinitely while the root domain remains
connected; granting socket or `udmabuf` access alone does not resolve that wait.
SCADA CPU clients can work even while GPU clients encounter this condition.

scadactl refuses to reuse an MPS daemon whose mode conflicts with the requested
mode, before binding drives. It does not restart that daemon. Stop its workloads
and daemon during a planned mode change, then apply the config again. Configs
sharing an MPS instance must request the same mode. If using custom
`CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY` values, pass them through
`sudo` when applying and use the same values for the server and client sessions;
`scadactl run` inherits these variables rather than discovering them from the
registry. Ensure directory access permits the intended clients. Separate pipe
directories name separate MPS instances and do not themselves establish that
concurrent use of the same GPU is supported.

## Running the tests

Run from the repository checkout root:

```shell
python3 -m pytest -q
```

The suite covers config validation (positive + documented
failure/warn case), plan/dry-run rendering, registry reconcile/atomic-write/locking,
client env generation + ordinal mapping + group resolution, alias<->id resolution,
UNVMe sysfs correlation, drive availability + device policy (allow/block decisions,
inventory annotation, apply gating), doctor checks, the config store + management
commands (list/show/edit/rm/import/compat) and apply-by-name, the interactive
wizard (build + edit), and CLI dispatch. It mocks NVML/sysfs/subprocess and points
`SCADA_RUNTIME_DIR` / `SCADA_CONFIG_DIR` / `SCADA_POLICY_FILE` at tmp dirs, so it
passes with no GPU and no root.

### Storage backends

The configuration wizard and `config add-domain` derive each domain's
`storage_backend` from its selected drives:
`kernel` for ordinary NVMe/UNVMe drives and `lender` for vfio-pci candidates.
All domains in one config must use the same backend. Install the matching server
package; the kernel and lender Debian server packages are mutually exclusive and
both provide `scada-swap-domain`.

Kernel candidates are ordinary NVMe controllers and UNVMe-bound devices. Lender
candidates are NVMe-class PCI functions bound to `vfio-pci`. Storage Lender does
not currently expose a discovery API, so a vfio candidate is treated as
optimistically usable rather than service-verified; a failed Lender launch is
reported normally by `apply`.

Kernel `apply` retains the bind/reclaim/unbind lifecycle and launches with
`/dev/unvme*` device nodes. Lender `apply` performs neither bind nor unbind and
launches the `scada-swap-domain` binary supplied by the installed Lender server
package with PCI BDFs. `STORAGE_LENDER_SOCKET` can be set in the existing global
or per-domain `server_env`; if unset, the runtime uses
`/run/storage-lender/api.sock`.

Before a non-dry-run Lender apply, `scadactl` requires that service socket to
exist. After every domain launch it waits (five seconds by default, configurable
with `SCADA_DOMAIN_READY_TIMEOUT_SEC`) for the expected SCADA domain socket before
publishing the domain to the registry; a timeout rolls the launch back.

For non-interactive Lender authoring, select a vfio-pci candidate by its PCI BDF
from `scadactl inventory`. Replace the example BDF below with that candidate:

```bash
scadactl config add-domain --name gpu0-lender --gpu gpu0 --drives 0000:3c:00.0 \
  -f lender.yaml
```

NVMe aliases use the enumerated controller name (`nvme0` for
`/dev/nvme0n1`); vfio-pci candidates use their PCI BDF. Controllers exposing
multiple namespaces are unsupported because their namespaces share the stable
serial and PCI identity stored in a configuration. Install `nvme-cli` for complete
namespace discovery rather than relying on the reduced sysfs fallback.

## Runtime interfaces

`backend/binding.py` uses the kernel backend's installed
`scada-load-modules --non-interactive` capability and parses machine-readable
`BOUND <pci> <unvmeN>` output. Targeted teardown uses the installed
`scada-unload-modules <pci...>` capability. These backend-specific commands are
provided by the UNVMe runtime rather than this project.

This project provides `scripts/manage-scada-domains`. System packages install it
at `/usr/sbin/manage-scada-domains`. `backend/launcher.py` invokes it once per
domain with a single-domain config and a log directory under
`/var/log/scada/<config>/<domain>/`.

Kernel mappings use `/dev/unvme*` device nodes. Storage Lender mappings use PCI
BDFs and skip kernel binding and unbinding. MPS control uses
`nvidia-cuda-mps-control` directly.

Managed profiler endpoints use
`/var/run/scada/scadad.profiler.gpu<N>.sock`. The single-domain `legacy_nsys`
mode uses `/tmp/scada_profiler_socket`.

UNVMe identity correlation reads
`/sys/class/unvme/unvme<N>/{serial,address,model}`. GPU selection uses NVML
indices as CUDA ordinals with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

## License

Copyright 2025-2026 NVIDIA Corporation and affiliates. The application code is
licensed under the [Apache License 2.0](LICENSE). Documentation-specific
third-party notices are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Contribution terms, security reporting, support, and project governance are
documented in [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md),
[SUPPORT.md](SUPPORT.md), and [GOVERNANCE.md](GOVERNANCE.md). Release changes
are recorded in [CHANGELOG.md](CHANGELOG.md).
