# Security policy

## Reporting a vulnerability

**Do not open a public issue, pull request, discussion, or merge request for a
security vulnerability.**

Report potential vulnerabilities to NVIDIA through one of these channels:

- [NVIDIA Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- Email: <psirt@nvidia.com>
- [NVIDIA PSIRT PGP key](https://www.nvidia.com/en-us/security/pgp-key)

Include the affected version or commit, vulnerability type, reproduction steps
or proof of concept, and potential impact. NVIDIA's coordinated disclosure
process is described in the
[PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/).

## Supported deployment assumptions

The supported security model requires:

- a dedicated Linux host whose GPUs and NVMe controllers are deliberately
  provisioned for SCADA;
- trusted administrators controlling root execution, configuration, device
  policy, installed backend capabilities, and environment overrides;
- secure ownership and permissions on `/etc/scada`, `/var/run/scada`,
  `/var/log/scada`, backend scripts, domain binaries, and socket directories;
- an NVIDIA driver, CUDA MPS, and storage backend installed from trusted sources;
  and
- workloads that follow the documented domain and device lifecycle.

Do not use boot devices, mounted filesystems, active RAID/LVM/swap members, or
data-bearing controllers as SCADA storage. Kernel-backed operation relies on the
installed UNVMe runtime's safety checks; `scadactl` does not reimplement them.

## Security boundaries

### Privileged operations

`apply`, `stop`, device-policy mutation, module binding, MPS control, and domain
process management can require root. Root execution trusts the installed
`scada-load-modules`, `scada-unload-modules`, `manage-scada-domains`,
`scada-swap-domain`, NVIDIA tools, and their absolute-path overrides. Protect
those files and do not set override variables from untrusted input.

### Configuration and persistent state

Configuration and policy files influence privileged device selection and process
environment. `/etc/scada` must be root-writable only. The runtime registry is
advisory state used for reconciliation; its lock, ownership, and atomic-write
properties must be preserved by custom packaging or service managers.

Per-domain `server_env` values are intentionally passed to the launched server.
Only trusted administrators may author configurations applied as root.

### Devices and backend capabilities

The kernel backend delegates bind and unbind safety to separately installed
UNVMe capabilities. The lender backend delegates controller isolation and DMA
safety to Storage Lender, VFIO/IOMMU, and their deployment policy. `scadactl`
coordinates these components but is not itself a hostile multi-tenant isolation
boundary.

### Local sockets and client admission

Access to SCADA and Storage Lender Unix sockets grants the operations exposed by
those services. Operators must use restrictive parent-directory, socket-owner,
group, and mode settings and admit only trusted local clients. Container device
and socket mappings remain part of the operator's security boundary.

### Process lifecycle

The domain manager verifies executable identity and Linux process start time
before signaling recorded PIDs. Force-stop behavior and process-name fallback
are administrative recovery mechanisms and must be used only after verifying the
target host state.

## Scope

Security reports may concern the CLI, configuration parsing, policy enforcement,
registry reconciliation, privileged helper invocation, process lifecycle,
filesystem permissions, Debian packaging, or interactions with supported storage
backends.
