# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config load/parse, schema validation, alias resolution, and authoring.

Implements configuration validation rules: hard failures that ``plan`` /
``apply`` reject, and warnings that authoring / planning surface.

Configs are authored with aliases on input but always persisted with stable
identifiers.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from scadactl import identity
from scadactl.models import (
    CONFIG_SCHEMA_VERSION,
    Domain,
    DriveRef,
    Inventory,
    Config,
    PROFILER_ENDPOINT_MODE_LEGACY_NSYS,
    PROFILER_ENDPOINT_MODES,
)


# --------------------------------------------------------------------------- #
# Validation result
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    """Outcome of validating a config: hard errors and soft warnings."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


# --------------------------------------------------------------------------- #
# Load / parse
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Config:
    """Load and parse a config YAML file into a :class:`Config`."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            # Surface malformed YAML as ValueError so callers' (OSError,
            # ValueError) handlers turn it into a clean error, not a traceback.
            raise ValueError(f"config {path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"config {path}: top-level YAML must be a mapping")
    return Config.from_dict(data)


def dump_config(config: Config) -> str:
    """Serialize a config to YAML (stable identifiers only)."""
    return yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False)


def save_config(config: Config, path: str) -> None:
    """Atomically write ``config`` as YAML to ``path`` with mode 0644 (world-readable, root-writable).

    Write to a temp file in the same directory, fsync, then ``os.replace`` --
    mirroring ``policy.save``/``registry._atomic_write`` -- so an interruption
    (e.g. Ctrl+C during ``config edit``/``init --force``/``import``) leaves either
    the full new content or the untouched original, never a truncated file.
    """
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(dump_config(config))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def summarize(config: Config, inventory: Optional[Inventory] = None) -> str:
    """Human-readable summary of a config (the shape the init wizard prints).

    Resolves GPU/drive aliases against ``inventory`` when available; otherwise
    falls back to the stable identifiers stored in the config.
    """
    lines: List[str] = [f"Config: {config.name}"]
    meta = []
    if config.created:
        meta.append(f"created {config.created}")
    if config.modified:
        meta.append(f"modified {config.modified}")
    if meta:
        lines.append("  " + "   ".join(meta))
    lines.append("")
    for d in config.domains:
        lines.append(d.name)
        g = identity.resolve_gpu(inventory, d.gpu_uuid) if inventory else None
        if g:
            lines.append(f"  GPU: {g.alias}, {g.name or ''}, {g.pci_bus_id or ''}")
        else:
            lines.append(f"  GPU: {d.gpu_uuid}, {d.gpu_pci_bus_id}")
        lines.append(f"  Storage backend: {d.storage_backend}")
        lines.append(f"  Socket group: {d.socket_group or '(root only)'}")
        for dr in d.drives:
            rd = (
                identity.resolve_drive(inventory, dr.serial or dr.pci_bus_id)
                if inventory
                else None
            )
            if rd:
                lines.append(f"  NVME: {rd.alias}, {rd.model or ''}, {rd.pci_bus_id or ''}")
            else:
                lines.append(f"  NVME: {dr.serial or ''}, , {dr.pci_bus_id or ''}")
        if d.server_env:
            ov = ", ".join(f"{k}={v}" for k, v in d.server_env.items())
            lines.append(f"  server_env (override): {ov}")
    lines.append("")
    for gname, members in config.domain_groups.items():
        lines.append(f"{gname} group: {', '.join(members)}")
    lines.append(f"MPS: {'enabled' if config.mps_enabled else 'disabled'}")
    if config.mps_enabled:
        lines.append(f"MPS multiuser: {config.mps_multiuser}")
    lines.append(f"Profiler endpoint mode: {config.profiler_endpoint_mode}")
    if config.server_env:
        env = ", ".join(f"{k}={v}" for k, v in config.server_env.items())
        lines.append(f"server_env: {env}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_config(
    config: Config,
    inventory: Optional[Inventory] = None,
    registry=None,
) -> ValidationResult:
    """Validate a single config against the configuration rules.

    HARD failures (``errors``):
      * unknown GPU UUID / GPU PCI bus id / drive serial / drive PCI bus id
        (only checkable when ``inventory`` is provided)
      * drive serial and PCI bus id that identify different inventory drives
      * alias input that cannot be resolved to a stable identifier
      * duplicate domain names
      * duplicate drive ownership across domains
      * the same drive listed more than once within one domain
      * duplicate GPU ownership across domains
      * domain with no drives
      * socket group that does not exist when the config requires one

    WARN (``warnings``):
      * alias input that resolves but would be unstable if written directly
      * resource (GPU/drive) already in use by a running domain not managed
        by this config (checkable when ``registry`` is provided)
    """
    res = ValidationResult()

    if config.profiler_endpoint_mode not in PROFILER_ENDPOINT_MODES:
        modes = ", ".join(sorted(PROFILER_ENDPOINT_MODES))
        res.errors.append(
            f"profiler.endpoint_mode must be one of: {modes} "
            f"(got {config.profiler_endpoint_mode!r})"
        )
    elif config.profiler_endpoint_mode == PROFILER_ENDPOINT_MODE_LEGACY_NSYS:
        if len(config.domains) != 1:
            res.errors.append(
                "profiler.endpoint_mode 'legacy_nsys' requires exactly one domain; "
                "use 'managed' for multi-GPU configurations"
            )
        if registry is not None:
            owners = sorted({
                d.config for d in registry.domains
                if d.config != config.name
                and d.profiler_endpoint_mode == PROFILER_ENDPOINT_MODE_LEGACY_NSYS
            })
            if owners:
                res.errors.append(
                    "profiler.endpoint_mode 'legacy_nsys' is already owned by active "
                    f"config(s): {', '.join(owners)}"
                )

    # Duplicate domain names.
    seen_names: set = set()
    for d in config.domains:
        if d.name in seen_names:
            res.errors.append(f"duplicate domain name: {d.name}")
        seen_names.add(d.name)

    # Duplicate drive ownership across domains (by serial and by PCI).
    # Also check for duplicate drives within one domain.
    drive_owner: Dict[str, str] = {}
    for d in config.domains:
        seen_in_domain: set = set()
        for dr in d.drives:
            # Compute tags for this drive
            tags_with_idents = []
            for kind, value in (("serial", dr.serial), ("pci", dr.pci_bus_id)):
                if not value:
                    continue
                tag = f"{kind}:{value}"
                tags_with_idents.append((tag, value))

            # Check for within-domain duplicate: if ANY tag is already in seen_in_domain
            if any(tag in seen_in_domain for tag, _ in tags_with_idents):
                # Report once using serial if set, else pci_bus_id
                drive_ident = dr.serial if dr.serial else dr.pci_bus_id
                res.errors.append(f"drive {drive_ident} listed more than once in domain {d.name}")
                continue  # Skip adding to drive_owner to avoid further processing

            # Check for cross-domain duplicate
            for tag, ident in tags_with_idents:
                if tag in drive_owner and drive_owner[tag] != d.name:
                    res.errors.append(
                        f"drive {ident} owned by multiple domains "
                        f"({drive_owner[tag]} and {d.name})"
                    )

            # Add to tracking
            for tag, _ in tags_with_idents:
                seen_in_domain.add(tag)
                drive_owner.setdefault(tag, d.name)

    # Duplicate GPU ownership across domains (by UUID and by PCI). A GPU maps to
    # exactly one swap domain/socket; two domains on one GPU collide at runtime.
    gpu_owner: Dict[str, str] = {}
    for d in config.domains:
        for kind, value in (("uuid", d.gpu_uuid), ("pci", d.gpu_pci_bus_id)):
            if not value:
                continue
            tag = f"{kind}:{value}"
            if tag in gpu_owner and gpu_owner[tag] != d.name:
                res.errors.append(
                    f"GPU {value} owned by multiple domains "
                    f"({gpu_owner[tag]} and {d.name})"
                )
            gpu_owner.setdefault(tag, d.name)

    # Debian packages install one server backend at a time, under the same
    # scada-swap-domain name. A config must therefore select one backend across
    # all of its domains.
    configured_backends = {d.storage_backend for d in config.domains}
    if len(configured_backends) > 1:
        res.errors.append(
            "config mixes storage backends; select either kernel NVMe drives or "
            "vfio-pci Storage Lender drives for the entire config"
        )

    if not isinstance(config.mps_multiuser, bool):
        res.errors.append("mps.multiuser must be a boolean")
    if config.mps_multiuser and not config.mps_enabled:
        res.errors.append("mps.multiuser requires mps.enabled: true")

    # Per-domain checks.
    for d in config.domains:
        if d.storage_backend not in ("kernel", "lender"):
            res.errors.append(
                f"domain {d.name}: invalid storage_backend '{d.storage_backend}'; "
                "expected 'kernel' or 'lender'"
            )
        # Domain with no drives.
        if not d.drives:
            res.errors.append(f"domain {d.name} has no drives")

        # Alias-as-stable-id check: identifiers must look stable, not aliases.
        if identity.is_alias_token(d.gpu_uuid) or identity.is_alias_token(d.gpu_pci_bus_id):
            res.warnings.append(
                f"domain {d.name}: GPU referenced by alias in a persisted field; "
                f"configs should store stable identifiers"
            )
        for dr in d.drives:
            if identity.is_alias_token(dr.serial) or identity.is_alias_token(dr.pci_bus_id):
                res.warnings.append(
                    f"domain {d.name}: drive referenced by alias in a persisted "
                    f"field; configs should store stable identifiers"
                )

        # Socket group existence (only when the config requires one).
        if d.socket_group:
            if not _group_exists(d.socket_group):
                res.errors.append(
                    f"domain {d.name}: socket group '{d.socket_group}' does not exist"
                )

    # Inventory-dependent identity checks.
    if inventory is not None:
        _validate_against_inventory(config, inventory, res)
        _warn_blocked_devices(config, inventory, res)

    # Registry-dependent resource-in-use warnings.
    if registry is not None:
        for entry in registry.configs:
            if (
                config.mps_enabled
                and entry.mps_enabled
                and entry.name != config.name
                and entry.name in registry.active_configs
                and entry.mps_multiuser != config.mps_multiuser
            ):
                res.errors.append(
                    f"active config '{entry.name}' requests a different MPS mode; "
                    "configs sharing MPS must use the same mps.multiuser setting"
                )
        _validate_against_registry(config, registry, res)

    # domain_groups reference real domains.
    for grp, members in config.domain_groups.items():
        for m in members:
            if m not in seen_names:
                res.errors.append(
                    f"domain_group '{grp}' references unknown domain '{m}'"
                )

    return res


def _validate_against_inventory(
    config: Config, inv: Inventory, res: ValidationResult
) -> None:
    gpu_uuids = {g.uuid for g in inv.gpus if g.uuid}
    gpu_pcis = {g.pci_bus_id for g in inv.gpus if g.pci_bus_id}
    gpu_by_uuid = {g.uuid: g for g in inv.gpus if g.uuid}
    gpu_by_pci = {g.pci_bus_id: g for g in inv.gpus if g.pci_bus_id}
    multi_ns = identity.multi_namespace_drives(inv.drives)
    drive_serials = {d.serial for d in inv.drives if d.serial}
    drive_pcis = {d.pci_bus_id for d in inv.drives if d.pci_bus_id}
    drive_by_serial = {d.serial: d for d in inv.drives if d.serial}
    drive_by_pci = {d.pci_bus_id: d for d in inv.drives if d.pci_bus_id}

    for d in config.domains:
        if d.gpu_uuid and d.gpu_uuid not in gpu_uuids:
            # Could be an unresolvable alias or simply unknown.
            if identity.is_alias_token(d.gpu_uuid):
                res.errors.append(
                    f"domain {d.name}: GPU alias '{d.gpu_uuid}' cannot be resolved"
                )
            else:
                res.errors.append(f"domain {d.name}: unknown GPU UUID {d.gpu_uuid}")
        if d.gpu_pci_bus_id and d.gpu_pci_bus_id not in gpu_pcis:
            if identity.is_alias_token(d.gpu_pci_bus_id):
                res.errors.append(
                    f"domain {d.name}: GPU alias '{d.gpu_pci_bus_id}' cannot be resolved"
                )
            else:
                res.errors.append(
                    f"domain {d.name}: unknown GPU PCI bus id {d.gpu_pci_bus_id}"
                )
        # Verify the GPU UUID and PCI bus id identify the same inventory GPU.
        # (Runtime selection follows the UUID, but apply/policy also resolve by
        # PCI, so a mismatched pair would silently target different GPUs.)
        g_by_uuid = gpu_by_uuid.get(d.gpu_uuid)
        g_by_pci = gpu_by_pci.get(d.gpu_pci_bus_id)
        if g_by_uuid is not None and g_by_pci is not None and g_by_uuid is not g_by_pci:
            res.errors.append(
                f"domain {d.name}: GPU UUID {d.gpu_uuid} and PCI {d.gpu_pci_bus_id} "
                f"identify different GPUs (UUID -> {g_by_uuid.pci_bus_id}, "
                f"PCI -> {g_by_pci.uuid}); a GPU's UUID and PCI bus id must match"
            )
        # Apply derives the launcher's GPU column (CUDA_VISIBLE_DEVICES) from the
        # resolved GPU's nvidia-smi index. Without one it would pass an empty
        # column, which the manager script rejects *after* the host was mutated --
        # so a config that cannot produce a complete launcher argument fails here.
        g_selected = g_by_uuid or g_by_pci
        if g_selected is not None and g_selected.smi_index is None:
            res.errors.append(
                f"domain {d.name}: GPU {d.gpu_uuid or d.gpu_pci_bus_id} has no "
                f"nvidia-smi index in the inventory; cannot derive "
                f"CUDA_VISIBLE_DEVICES for the launcher"
            )
        for dr in d.drives:
            if dr.serial and dr.serial not in drive_serials:
                if identity.is_alias_token(dr.serial):
                    res.errors.append(
                        f"domain {d.name}: drive alias '{dr.serial}' cannot be resolved"
                    )
                else:
                    res.errors.append(
                        f"domain {d.name}: unknown drive serial {dr.serial}"
                    )
            if dr.pci_bus_id and dr.pci_bus_id not in drive_pcis:
                if identity.is_alias_token(dr.pci_bus_id):
                    res.errors.append(
                        f"domain {d.name}: drive alias '{dr.pci_bus_id}' cannot be resolved"
                    )
                else:
                    res.errors.append(
                        f"domain {d.name}: unknown drive PCI bus id {dr.pci_bus_id}"
                    )
            # Verify that serial and PCI bus id identify the same inventory drive.
            s_drv = drive_by_serial.get(dr.serial)
            p_drv = drive_by_pci.get(dr.pci_bus_id)
            if s_drv is not None and p_drv is not None and s_drv is not p_drv:
                res.errors.append(
                    f"domain {d.name}: drive serial {dr.serial} and PCI {dr.pci_bus_id} "
                    f"identify different drives (serial -> {s_drv.pci_bus_id}, "
                    f"PCI -> {p_drv.serial}); a drive's serial and PCI bus id must match"
                )
            selected = s_drv or p_drv
            if selected is not None and selected.pci_bus_id in multi_ns:
                res.errors.append(
                    f"domain {d.name}: drive {selected.pci_bus_id} exposes multiple "
                    f"namespaces ({', '.join(multi_ns[selected.pci_bus_id])}); "
                    f"scadactl supports one namespace per controller"
                )
            selected_backend = selected.storage_backend if selected is not None else None
            # Existing hand-authored configs and external callers may construct
            # an Inventory without the new field; retain kernel compatibility.
            if selected_backend is None:
                selected_backend = "kernel"
            if selected is not None and selected_backend != d.storage_backend:
                res.errors.append(
                    f"domain {d.name}: drive {selected.alias} ({selected.pci_bus_id}) "
                    f"is a {selected_backend} storage candidate, "
                    f"not {d.storage_backend}"
                )


def iter_blocked_selections(config: Config, inv: Inventory):
    """Yield ``(domain_name, kind, alias, reason)`` for each config-selected
    device the host policy blocks. ``kind`` is ``"GPU"`` or ``"drive"``.

    Shared by ``_warn_blocked_devices`` (authoring: warns) and
    ``apply._blocked_selections`` (apply: hard-gates without ``--force``) so
    the two do not drift.
    """
    for d in config.domains:
        g = identity.resolve_gpu(inv, d.gpu_uuid or d.gpu_pci_bus_id or "")
        if g is not None and g.blocked:
            yield d.name, "GPU", g.alias, g.block_reason
        for dr in d.drives:
            rd = identity.resolve_drive(inv, dr.serial or dr.pci_bus_id or "")
            if rd is not None and rd.blocked:
                yield d.name, "drive", rd.alias, rd.block_reason


def _warn_blocked_devices(config: Config, inv: Inventory, res: ValidationResult) -> None:
    """Warn when a config selects a device the host policy blocks (apply gates)."""
    for domain, kind, alias, reason in iter_blocked_selections(config, inv):
        res.warnings.append(
            f"domain {domain}: {kind} {alias} is blocked by policy "
            f"({reason}); apply will refuse it without --force"
        )


def _validate_against_registry(config: Config, registry, res: ValidationResult) -> None:
    """Warn when a GPU/drive is already used by a domain we do not own.

    Also hard-errors when a domain *name* is already owned by a different
    active config: domain names are globally unique in the registry, and
    ``apply`` would otherwise crash midway through reconciling the host (see
    ``registry.reconcile_add_config``). Reapplying the SAME config is not
    flagged.
    """
    used_gpus: Dict[str, str] = {}
    used_drives: Dict[str, str] = {}
    name_owner: Dict[str, str] = {}
    for rd in registry.domains:
        if rd.config == config.name:
            continue
        if rd.gpu_uuid:
            used_gpus[rd.gpu_uuid] = rd.name
        for drv in rd.drives:
            if drv.serial:
                used_drives[drv.serial] = rd.name
        name_owner[rd.name] = rd.config

    for d in config.domains:
        if d.name in name_owner:
            res.errors.append(
                f"domain {d.name}: name already owned by active config "
                f"'{name_owner[d.name]}' (domain names are globally unique)"
            )
        if d.gpu_uuid in used_gpus:
            res.warnings.append(
                f"domain {d.name}: GPU {d.gpu_uuid} already in use by running "
                f"domain {used_gpus[d.gpu_uuid]}"
            )
        for dr in d.drives:
            if dr.serial in used_drives:
                res.warnings.append(
                    f"domain {d.name}: drive {dr.serial} already in use by running "
                    f"domain {used_drives[dr.serial]}"
                )


def _group_exists(group: str) -> bool:
    """Return True if a unix group named ``group`` exists (best-effort)."""
    try:
        import grp as _grp

        _grp.getgrnam(group)
        return True
    except KeyError:
        return False
    except Exception:
        # On platforms without grp, do not hard-fail; treat as present.
        return True


# --------------------------------------------------------------------------- #
# Composition validation
# --------------------------------------------------------------------------- #
def validate_composition(configs: List[Config]) -> ValidationResult:
    """Validate that a set of configs compose without colliding.

    Configs that need the same GPU, drive, or socket namespace (a globally
    unique domain name) are mutually exclusive at runtime.
    """
    res = ValidationResult()
    name_owner: Dict[str, str] = {}
    gpu_owner: Dict[str, str] = {}   # keyed by UUID *and* PCI bus id
    drive_owner: Dict[str, str] = {}  # keyed by serial *and* PCI bus id
    modes = {p.mps_multiuser for p in configs if p.mps_enabled}
    if len(modes) > 1:
        res.errors.append("configs request incompatible MPS modes; use the same mps.multiuser setting")

    def _claim(owner: Dict[str, str], keys, kind: str, config_name: str) -> None:
        """Record a device's ownership by each of its stable ids and report the
        first collision with a different config. Both UUID/serial and PCI bus id
        are stable selectors, so a config that pins a device by PCI only still
        collides with another config selecting the same device."""
        reported = False
        for key in keys:
            if not key:
                continue
            other = owner.get(key)
            if other is not None and other != config_name and not reported:
                res.errors.append(
                    f"composed configs collide on {kind} {key} "
                    f"({other} and {config_name})"
                )
                reported = True
            owner.setdefault(key, config_name)

    for p in configs:
        for d in p.domains:
            if d.name in name_owner and name_owner[d.name] != p.name:
                res.errors.append(
                    f"composed configs collide on domain name '{d.name}' "
                    f"({name_owner[d.name]} and {p.name})"
                )
            name_owner.setdefault(d.name, p.name)

            _claim(gpu_owner, (d.gpu_uuid, d.gpu_pci_bus_id), "GPU", p.name)
            for dr in d.drives:
                _claim(drive_owner, (dr.serial, dr.pci_bus_id), "drive", p.name)
    return res


# --------------------------------------------------------------------------- #
# Alias resolution (authoring: accept aliases, write stable ids)
# --------------------------------------------------------------------------- #
def resolve_domain_aliases(domain: Domain, inv: Inventory) -> Domain:
    """Resolve any alias references in a domain to stable identifiers.

    Raises ``ValueError`` if a token cannot be resolved.
    """
    gpu = None
    for token in (domain.gpu_uuid, domain.gpu_pci_bus_id):
        if token:
            gpu = identity.resolve_gpu(inv, token)
            if gpu:
                break
    if gpu is None:
        raise ValueError(
            f"domain {domain.name}: cannot resolve GPU "
            f"'{domain.gpu_uuid or domain.gpu_pci_bus_id}'"
        )
    if not (gpu.uuid or gpu.pci_bus_id):
        raise ValueError(
            f"domain {domain.name}: resolved GPU has no stable identifier "
            f"(UUID or PCI bus id) to persist"
        )

    resolved_drives: List[DriveRef] = []
    for dr in domain.drives:
        token = dr.serial or dr.pci_bus_id
        drive = identity.resolve_drive(inv, token)
        if drive is None:
            raise ValueError(f"domain {domain.name}: cannot resolve drive '{token}'")
        if not (drive.serial or drive.pci_bus_id):
            raise ValueError(
                f"domain {domain.name}: resolved drive '{token}' has no stable "
                f"identifier (serial or PCI bus id) to persist"
            )
        resolved_drives.append(
            DriveRef(serial=drive.serial or "", pci_bus_id=drive.pci_bus_id or "")
        )

    return Domain(
        name=domain.name,
        gpu_uuid=gpu.uuid or "",
        gpu_pci_bus_id=gpu.pci_bus_id or "",
        storage_backend=domain.storage_backend,
        drives=resolved_drives,
        server_env=dict(domain.server_env),
        socket_group=domain.socket_group,
    )


# --------------------------------------------------------------------------- #
# Authoring: init / add-domain
# --------------------------------------------------------------------------- #
def empty_config(name: str = "scadactl") -> Config:
    """A schema skeleton with global env defaults and no domains."""
    return Config(
        name=name,
        mps_enabled=False,
        server_env={
            "SCADA_NUM_QUEUE_PAIRS": "16",
            "SCADA_QUEUE_SIZE_SQ_GPU": "1024",
            "SCADA_QUEUE_SIZE_CQ_GPU": "1024",
        },
        domains=[],
        domain_groups={},
    )


def seed_config_from_resources(
    inv: Inventory,
    name: str,
    gpu_tokens: List[str],
    drive_tokens: List[str],
) -> Config:
    """Seed a config from explicit ``--gpus`` / ``--drives`` selections.

    Pairs GPUs and drives positionally into one domain per GPU, resolving every
    token to a stable identifier (raises ``ValueError`` on an unresolved token).
    """
    config = empty_config(name)
    resolved_gpus = []
    for t in gpu_tokens:
        g = identity.resolve_gpu(inv, t)
        if g is None:
            raise ValueError(f"cannot resolve GPU '{t}'")
        if not (g.uuid or g.pci_bus_id):
            raise ValueError(
                f"GPU '{t}' has no stable identifier (UUID or PCI bus id) to persist"
            )
        resolved_gpus.append(g)
    resolved_drives = []
    for t in drive_tokens:
        d = identity.resolve_drive(inv, t)
        if d is None:
            raise ValueError(f"cannot resolve drive '{t}'")
        if not (d.serial or d.pci_bus_id):
            raise ValueError(
                f"drive '{t}' has no stable identifier (serial or PCI bus id) to persist"
            )
        resolved_drives.append(d)

    # Distribute drives round-robin across the selected GPUs.
    domains: List[Domain] = []
    for i, g in enumerate(resolved_gpus):
        mine = resolved_drives[i :: len(resolved_gpus)] if resolved_gpus else []
        domains.append(
            Domain(
                name=f"{g.alias}_domain",
                gpu_uuid=g.uuid or "",
                gpu_pci_bus_id=g.pci_bus_id or "",
                drives=[
                    DriveRef(serial=d.serial or "", pci_bus_id=d.pci_bus_id or "")
                    for d in mine
                ],
            )
        )
    config.domains = domains
    if domains:
        config.domain_groups = {"all": [d.name for d in domains]}
    return config


def add_domain(
    config: Config,
    inv: Inventory,
    name: str,
    gpu_token: str,
    drive_tokens: List[str],
    server_env: Optional[Dict[str, str]] = None,
    socket_group: Optional[str] = None,
) -> Config:
    """Add a domain (accepting aliases), writing STABLE identifiers.

    Raises ``ValueError`` on an unresolved token or a duplicate domain name.
    """
    if not drive_tokens:
        raise ValueError("a domain requires at least one drive")
    if any(d.name == name for d in config.domains):
        raise ValueError(f"domain name already exists: {name}")
    raw = Domain(
        name=name,
        gpu_uuid=gpu_token,
        gpu_pci_bus_id=gpu_token,
        drives=[DriveRef(serial=t, pci_bus_id=t) for t in drive_tokens],
        server_env=server_env or {},
        socket_group=socket_group,
    )
    resolved = resolve_domain_aliases(raw, inv)
    selected_drives = [
        identity.resolve_drive(inv, dr.serial or dr.pci_bus_id)
        for dr in resolved.drives
    ]
    selected_backends = {
        drive.storage_backend or "kernel" for drive in selected_drives if drive is not None
    }
    if len(selected_backends) != 1:
        raise ValueError(
            "selected drives use different storage backends; choose either "
            "kernel NVMe drives or vfio-pci Storage Lender drives"
        )
    selected_backend = next(iter(selected_backends))
    configured_backends = {d.storage_backend for d in config.domains}
    if configured_backends and selected_backend not in configured_backends:
        raise ValueError(
            "config already uses the " + next(iter(configured_backends)) +
            " backend; installed server packages are mutually exclusive"
        )
    resolved.storage_backend = selected_backend
    config.domains.append(resolved)
    return config
