# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply: reconcile the host to a config (privileged) and publish the registry.

Order: validate -> bind selected drives -> start MPS (if requested) -> launch
one domain at a time -> publish registry. Idempotent reconcile. ``--dry-run``
prints planned actions without touching the host. All privileged operations go
through the ``backend`` wrappers.

The hardware/root-dependent steps degrade gracefully and are guarded so the
module imports and the planning/registry logic is testable without a GPU/root.
"""

from __future__ import annotations

import functools
import os
import stat
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scadactl import identity as identity_mod
from scadactl import config as config_mod
from scadactl import reconcile as reconcile_mod
from scadactl import registry as registry_mod
from scadactl.backend import binding, launcher, lender, mps
from scadactl.models import (
    Inventory,
    Config,
    Registry,
    RegistryDomain,
    RegistryDrive,
    PROFILER_ENDPOINT_MODE_LEGACY_NSYS,
)


@dataclass
class ApplyResult:
    config: str
    actions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    registry: Optional[Registry] = None
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors


DEFAULT_DOMAIN_READY_TIMEOUT_SEC = 5.0
DEFAULT_STORAGE_LENDER_SOCKET = "/run/storage-lender/api.sock"


def _remove_managed_profiler_socket(domain: RegistryDomain) -> bool:
    """Remove an orphaned profiler socket after its owning domain is stopped.

    The domain process normally unlinks this endpoint, but a managed profiler
    shutdown can leave the Unix socket pathname behind. Only remove the
    per-domain managed endpoint recorded in the registry; legacy Nsight uses a
    shared endpoint and is deliberately left alone.
    """
    if (
        domain.profiler_endpoint_mode == PROFILER_ENDPOINT_MODE_LEGACY_NSYS
        or not domain.profiler_socket
    ):
        return False
    try:
        if not stat.S_ISSOCK(os.lstat(domain.profiler_socket).st_mode):
            return False
        os.unlink(domain.profiler_socket)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        # The process is stopped and registry teardown must still complete.
        return False


def _storage_lender_socket(domain, config: Config) -> str:
    return domain.resolved_server_env(config.server_env).get(
        "STORAGE_LENDER_SOCKET", DEFAULT_STORAGE_LENDER_SOCKET
    )


def _socket_signature(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino, st.st_mtime_ns


def _domain_ready_timeout() -> float:
    raw = os.environ.get("SCADA_DOMAIN_READY_TIMEOUT_SEC")
    try:
        return max(0.1, float(raw)) if raw is not None else DEFAULT_DOMAIN_READY_TIMEOUT_SEC
    except ValueError:
        return DEFAULT_DOMAIN_READY_TIMEOUT_SEC


def _wait_for_domain_socket(path: str, prior_signature, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        signature = _socket_signature(path)
        if signature is not None and signature != prior_signature:
            return True
        time.sleep(0.1)
    signature = _socket_signature(path)
    return signature is not None and signature != prior_signature


def build_registry_domains(
    config: Config,
    inventory: Inventory,
    unvme_by_pci: Optional[dict] = None,
) -> List[RegistryDomain]:
    """Build live registry domain records for a config.

    ``unvme_by_pci`` maps a drive PCI bus id to its assigned ``/dev/unvmeN``
    (recovered after binding). When absent (e.g. dry-run/plan), ``unvme_device``
    is left ``None``.

    Every selector form validation accepts is resolved through the inventory, so
    execution and validation agree: a GPU pinned by PCI **only** still yields a
    host CUDA index and UUID, and a drive pinned by serial **only** still yields
    the PCI BDF that binding and the launcher require. Resolved identities are
    what gets published, so the registry record is always complete.
    """
    unvme_by_pci = unvme_by_pci or {}
    out: List[RegistryDomain] = []
    for dom in config.domains:
        # Resolve by whichever stable id the config carries (UUID or PCI).
        gpu = identity_mod.resolve_gpu(inventory, dom.gpu_uuid or dom.gpu_pci_bus_id)
        gpu_alias = gpu.alias if gpu else None
        # Host CUDA index comes from the resolved GPU, so a PCI-only selector
        # still produces a usable CUDA_VISIBLE_DEVICES for the manager script.
        idx = gpu.smi_index if gpu else None
        host_cvd = str(idx) if idx is not None else ""
        gpu_uuid = (gpu.uuid if gpu else None) or dom.gpu_uuid
        gpu_pci = (gpu.pci_bus_id if gpu else None) or dom.gpu_pci_bus_id
        socket = registry_mod.socket_for_domain(
            gpu_alias or (gpu_pci or gpu_uuid).replace(":", "").replace(".", "")
        )
        profiler_socket = (
            registry_mod.LEGACY_NSYS_PROFILER_SOCKET
            if config.profiler_endpoint_mode == PROFILER_ENDPOINT_MODE_LEGACY_NSYS
            else registry_mod.profiler_socket_for_domain(
                gpu_alias or (gpu_pci or gpu_uuid).replace(":", "").replace(".", "")
            )
        )
        drives: List[RegistryDrive] = []
        for dr in dom.drives:
            inv_drive = identity_mod.resolve_drive(inventory, dr.serial or dr.pci_bus_id)
            # Prefer the inventory's PCI: a serial-only DriveRef carries none, and
            # without it the drive would never be bound and the launcher would get
            # an empty device argument.
            pci = (inv_drive.pci_bus_id if inv_drive else None) or dr.pci_bus_id
            drives.append(
                RegistryDrive(
                    linux_device_before_bind=inv_drive.linux_device if inv_drive else None,
                    serial=dr.serial or (inv_drive.serial if inv_drive else ""),
                    model=inv_drive.model if inv_drive else None,
                    namespace=inv_drive.namespace if inv_drive else None,
                    lba_size=inv_drive.lba_size if inv_drive else None,
                    pci_bus_id=pci,
                    pcie=inv_drive.pcie if inv_drive else None,
                    unvme_device=unvme_by_pci.get(pci),
                )
            )
        out.append(
            RegistryDomain(
                name=dom.name,
                config=config.name,
                gpu_alias=gpu_alias,
                gpu_uuid=gpu_uuid,
                gpu_pci_bus_id=gpu_pci,
                host_cuda_visible_devices=host_cvd,
                socket=socket,
                socket_group=dom.socket_group,
                profiler_socket=profiler_socket,
                profiler_endpoint_mode=config.profiler_endpoint_mode,
                storage_backend=dom.storage_backend,
                server_env=dom.resolved_server_env(config.server_env),
                drives=drives,
            )
        )
    return out


def _host_locked(fn):
    """Run a privileged host operation under the operation-wide host lock.

    ``apply``/``stop`` hold ``registry.host_lock()`` across their whole
    ``validate -> mutate -> publish`` span so two concurrent operations serialize
    (see ``registry.host_lock``). A dry-run mutates nothing and takes no lock, so
    previews never block on -- or serialize behind -- a running apply.
    """
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        if kwargs.get("dry_run", False):
            return fn(*args, **kwargs)
        with registry_mod.host_lock():
            return fn(*args, **kwargs)
    return _wrapped


@_host_locked
def apply_config(
    config: Config,
    inventory: Inventory,
    dry_run: bool = False,
    force: bool = False,
) -> ApplyResult:
    """Reconcile the host to ``config`` and publish the registry."""
    result = ApplyResult(config=config.name, dry_run=dry_run)

    # 1) Validate (hard-fail rules block apply).
    existing = registry_mod.read_registry()
    vres = config_mod.validate_config(config, inventory=inventory, registry=existing)
    for w in vres.warnings:
        result.actions.append(f"warning: {w}")
    if not vres.ok:
        result.errors.extend(vres.errors)
        return result

    # 1b) Device policy gate: refuse blocked devices unless forced.
    blocked = _blocked_selections(config, inventory)
    if blocked and not force:
        result.errors.extend(blocked)
        result.errors.append("use --force to override device policy")
        return result
    for b in blocked:  # forced through: record for the audit trail
        result.actions.append(f"warning: {b} (overridden by --force)")

    # 2) Reconcile the delta. Classify each desired domain against the registry
    # (the source of truth for what is managed) and inventory (actual host state).
    # A conflict with another active config, or an in-place change to this active
    # config, blocks the apply before any mutation; an unchanged re-apply is a
    # no-op.
    desired = build_registry_domains(config, inventory)
    plan = reconcile_mod.plan_reconciliation(config, inventory, existing, desired)
    if not plan.ok:
        # Conflict with another active config, or an in-place change to this
        # active config -- refuse before touching anything.
        result.errors.extend(plan.errors)
        return result
    for dp in plan.domains:
        for reason in dp.reasons:
            result.actions.append(f"plan: {dp.name}: {reason}")
    if plan.reclaim_pci():
        # Loud, up-front: we are taking over resources currently set up on the
        # host in an unmanaged way (dry-run must show this before it happens).
        result.actions.append(
            "WARNING: reclaiming unmanaged resources currently set up on the host "
            f"-- drive(s) {', '.join(plan.reclaim_pci())} will be torn down and rebuilt"
        )
    if config.mps_enabled and not dry_run:
        error = mps.compatibility_error(config.mps_multiuser)
        if error:
            result.errors.append(error)
            return result
    if not plan.actionable():
        result.actions.append(
            f"config {config.name} already reconciled; {len(plan.domains)} "
            f"domain(s) running, nothing to do"
        )
        result.registry = existing
        return result

    actionable = {dp.name for dp in plan.actionable()}
    bind_pci = plan.bind_pci()
    reclaim_pci = plan.reclaim_pci()

    lender_sockets = {
        _storage_lender_socket(dom, config)
        for dom in config.domains
        if dom.name in actionable and dom.storage_backend == "lender"
    }
    unavailable_lender_sockets = sorted(
        sock for sock in lender_sockets if not lender.is_unix_socket(sock)
    )
    if unavailable_lender_sockets:
        detail = ", ".join(unavailable_lender_sockets)
        if dry_run:
            result.actions.append(
                "preflight: Storage Lender endpoint(s) unavailable or not Unix sockets "
                "(would fail apply): " + detail
            )
        else:
            result.errors.append(
                "Storage Lender endpoint unavailable or not a Unix socket: " + detail +
                "; start storage-lender-server or set STORAGE_LENDER_SOCKET"
            )
            return result

    # Track this invocation's side effects so a failure can be rolled back.
    launched: List[Tuple[str, str, str]] = []   # (config, domain, backend) (re)built OK here
    release_pci: List[str] = []            # drives THIS apply bound
    started_mps = False                    # True only if THIS apply started MPS

    # 3) Free unmanaged orphan drives so they can be rebound cleanly. This runs
    # before we bind or launch anything, so a failure here has nothing of ours to
    # roll back -- report and stop.
    if reclaim_pci:
        rres = binding.unbind_drives(reclaim_pci, dry_run=dry_run)
        result.actions.append(
            f"reclaim {len(reclaim_pci)} unmanaged drive(s) via "
            f"{binding.unload_script_path()}"
        )
        if not dry_run and not rres.ok:
            result.errors.append(
                f"reclaim of unmanaged drive(s) failed: {_fail_detail(rres)}"
            )
            return result

    # 4) Bind the drives this apply needs (only those not already correctly bound).
    unvme_by_pci: dict = {}
    if bind_pci:
        bind_res = binding.bind_drives(bind_pci, dry_run=dry_run)
        result.actions.append(
            f"bind {len(bind_pci)} drive(s) via {binding.load_script_path()} "
            f"--non-interactive --print-result"
        )
        if not dry_run:
            # Roll back ONLY the drives the load script authoritatively confirmed
            # it bound (its BOUND lines) -- never the full requested set. A drive
            # the script refused for safety (mount/RAID/LVM/swap) is still on nvme;
            # unbinding it on rollback would detach a live device (e.g. the mounted
            # filesystem that caused the rejection), so the old fallback to the whole
            # requested set was unsafe. If the script confirmed nothing, we release
            # nothing: a stale unvme bind is recoverable via stop/manual unload,
            # whereas detaching a mounted drive is not.
            release_pci = list(bind_res.bound_pci)
        if not dry_run and not bind_res.ok:
            result.errors.append(f"drive binding failed: {_fail_detail(bind_res)}")
            return _fail_with_rollback(
                result, existing, config, launched, release_pci, started_mps=started_mps
            )
        # The load script suppresses per-drive bind errors (`|| true`) and exits 0
        # even if some drives never joined unvme, so a zero exit is not proof of a
        # complete bind. Its authoritative BOUND set must therefore cover every
        # requested drive; if any is missing -- INCLUDING the all-failed case where
        # nothing bound -- refuse rather than launch a silently under-provisioned
        # domain (the launcher accepts any non-empty drive count, so a short bind
        # would not be caught there). The script restores any un-bound drive to
        # nvme, so the missing ones are already off unvme and need no rollback.
        if not dry_run and bind_pci:
            missing = [p for p in bind_pci if p not in set(bind_res.bound_pci)]
            if missing:
                result.errors.append(
                    f"drive binding incomplete: {len(bind_pci) - len(missing)} of "
                    f"{len(bind_pci)} drive(s) joined unvme; missing "
                    f"{', '.join(missing)} (check dmesg / drive eligibility)"
                )
                return _fail_with_rollback(
                    result, existing, config, launched, release_pci, started_mps=started_mps
                )
        if not dry_run:
            unvme_by_pci = bind_res.unvme_by_pci or {}
    if not dry_run:
        # Self-heal domains reuse a drive that is already bound; recover its
        # /dev/unvmeN from the driver's sysfs identity. Freshly-bound drives from
        # the load script's authoritative mapping take precedence.
        unvme_by_pci = {**_recover_unvme_mapping(inventory), **unvme_by_pci}

    # 5) Start MPS if requested. Record whether we actually brought it up (it was
    # not already running) so rollback stops only a daemon THIS apply started.
    if config.mps_enabled:
        mps_was_running = mps.is_running()
        mres = mps.start(dry_run=dry_run, multiuser=config.mps_multiuser)
        result.actions.append(f"ensure MPS via {' '.join(mps.start_command(config.mps_multiuser))}")
        if not dry_run and not mres.ok:
            result.errors.append(f"MPS start failed: {_fail_detail(mres)}")
            return _fail_with_rollback(
                result, existing, config, launched, release_pci, started_mps=started_mps
            )
        if not dry_run and mres.ok and not mps_was_running:
            started_mps = True

    # 6) (Re)launch only the domains that need it; leave running ones untouched.
    # The published registry merges each no-op domain's existing record with the
    # freshly (re)built records, so a partial reconcile never drops a live domain.
    reg_domains = build_registry_domains(config, inventory, unvme_by_pci)
    recorded = {d.name: d for d in existing.domains if d.config == config.name}
    publish_domains: List[RegistryDomain] = []
    rebuilt: List[RegistryDomain] = []
    for dom, rd in zip(config.domains, reg_domains):
        if dom.name not in actionable:
            publish_domains.append(recorded.get(dom.name, rd))  # keep it untouched
            continue
        rebuilt.append(rd)
        prior_socket_signature = _socket_signature(rd.socket)
        if rd.storage_backend == "kernel":
            storage_devices = [d.unvme_device for d in rd.drives if d.unvme_device]
            # Dry-run deliberately skips binding, so no /dev/unvmeN mapping
            # exists yet. Preserve a useful plan by showing the stable BDFs;
            # a real apply continues to require the recovered UNVMe mapping.
            if dry_run and not storage_devices:
                storage_devices = [d.pci_bus_id for d in rd.drives if d.pci_bus_id]
        else:
            storage_devices = [d.pci_bus_id for d in rd.drives if d.pci_bus_id]
        launch_kwargs = dict(
            config=config.name,
            domain=dom.name,
            host_cuda_index=rd.host_cuda_visible_devices,
            unvme_devices=storage_devices,
            resolved_server_env=rd.server_env,
            profiler_socket=rd.profiler_socket,
            socket_group=dom.socket_group,
            dry_run=dry_run,
        )
        if rd.storage_backend == "lender":
            launch_kwargs["storage_backend"] = "lender"
        lres = launcher.launch_domain(**launch_kwargs)
        result.actions.append(
            f"launch domain {dom.name} (CUDA {rd.host_cuda_visible_devices}, "
            f"config '{lres.config_line}', log-dir {lres.log_dir})"
        )
        if not dry_run:
            rd.healthy = lres.ok
            rd.pid = lres.pid
        if not dry_run and not lres.ok:
            result.errors.append(
                f"domain {dom.name} launch failed: {_fail_detail(lres)}"
            )
            return _fail_with_rollback(
                result, existing, config, launched, release_pci,
                reg_domains=rebuilt, started_mps=started_mps,
            )
        ready_timeout = _domain_ready_timeout()
        if not dry_run and not _wait_for_domain_socket(
            rd.socket, prior_socket_signature, ready_timeout
        ):
            result.errors.append(
                f"domain {dom.name} did not become ready within {ready_timeout:g}s: "
                f"expected socket {rd.socket}; verify the selected "
                f"{rd.storage_backend} backend and server package"
            )
            launched.append((config.name, dom.name, rd.storage_backend))
            return _fail_with_rollback(
                result, existing, config, launched, release_pci,
                reg_domains=rebuilt, started_mps=started_mps,
            )
        if not dry_run and lres.ok:
            launched.append((config.name, dom.name, rd.storage_backend))
        publish_domains.append(rd)

    # 7) Publish the registry (reconcile this config's entries). The real write
    # does read + reconcile + write in one locked critical section (see
    # registry.update_registry) so a concurrent apply/stop cannot drop our
    # domains -- or have theirs dropped by us -- via a stale snapshot.
    def _publish(current):
        return registry_mod.reconcile_add_config(
            current,
            config.name,
            publish_domains,
            domain_groups=config.domain_groups,
            mps_enabled=config.mps_enabled,
            mps_multiuser=config.mps_multiuser,
        )

    try:
        if dry_run:
            result.registry = _publish(existing)
            result.actions.append(
                f"publish {registry_mod.registry_path()} (skipped: dry-run)"
            )
        else:
            result.registry = registry_mod.update_registry(_publish)
            result.actions.append(f"published {registry_mod.registry_path()}")
    except Exception as exc:  # noqa: BLE001 - any publish failure after mutation must roll back
        # Drives/MPS/domains are already live by now. The publish can still fail --
        # a fresh-state cross-config domain-name collision re-checked inside the
        # lock, or a lock/open/write/fsync error -- so roll this invocation's
        # changes back rather than orphan live resources with no registry entry.
        # (Dry-run mutated nothing, so just report.)
        result.errors.append(str(exc))
        if dry_run:
            return result
        return _fail_with_rollback(
            result, existing, config, launched, release_pci,
            reg_domains=rebuilt, started_mps=started_mps,
        )
    return result


def _blocked_selections(config: Config, inventory: Inventory) -> List[str]:
    """Messages for any config-selected device the host policy blocks."""
    return [
        f"domain {domain}: {kind} {alias} is blocked by policy ({reason})"
        for domain, kind, alias, reason in config_mod.iter_blocked_selections(config, inventory)
    ]


def planned_commands(config: Config, inventory: Optional[Inventory]) -> List[str]:
    """The literal backend commands a real apply would run, for ``--dry-run
    --commands``. Drives are shown by PCI (their ``/dev/unvmeN`` paths are only
    known after binding), matching what apply passes pre-correlation.

    The per-domain config is shown inline as ``--config <(echo '…')`` to convey
    its contents; apply writes that line to a temp file rather than running this
    string verbatim, so the output is illustrative, not copy-paste-runnable.
    """
    # A None inventory (best-effort load failed) still yields a usable preview:
    # identity resolution just returns nothing and drives are shown by PCI.
    inventory = inventory or Inventory()
    cmds: List[str] = []
    all_pci = [
        dr.pci_bus_id for dom in config.domains if dom.storage_backend == "kernel"
        for dr in dom.drives
    ]
    if all_pci:
        cmds.append(
            f"{binding.load_script_path()} --non-interactive " + " ".join(all_pci)
        )
    if config.mps_enabled:
        cmds.append(
            f"CUDA_MPS_PIPE_DIRECTORY={mps.pipe_directory()} "
            f"CUDA_MPS_LOG_DIRECTORY={mps.log_directory()} {' '.join(mps.start_command(config.mps_multiuser))}"
        )
    reg_domains = build_registry_domains(config, inventory)
    for dom, rd in zip(config.domains, reg_domains):
        log_dir = launcher.domain_log_dir(config.name, dom.name)
        config_line = launcher.build_config_line(
            rd.host_cuda_visible_devices, [d.pci_bus_id for d in rd.drives]
        )
        env = dict(rd.server_env)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env_str = " ".join(f"{k}={v}" for k, v in env.items())
        group = f" --group {dom.socket_group}" if dom.socket_group else ""
        cmds.append(
            f"{env_str} {launcher.manage_script_path()} launch "
            f"--storage-backend {rd.storage_backend} "
            f"--domain-binary {launcher.domain_binary(rd.storage_backend)} "
            f"--config <(echo '{config_line}') --log-dir {log_dir}{group}"
        )
    return cmds


def _fail_detail(res) -> str:
    """Best-effort failure detail from a backend result.

    Backend scripts (e.g. scada-load-modules) often print their error to stdout
    rather than stderr, so surface whichever is non-empty, and always include the
    exit code so a silent non-zero return is never reported as a blank error.
    """
    stderr = (getattr(res, "stderr", "") or "").strip()
    stdout = (getattr(res, "stdout", "") or "").strip()
    detail = stderr or stdout or "(no output)"
    rc = getattr(res, "returncode", None)
    return f"{detail} [exit {rc}]" if rc is not None else detail


def _recover_unvme_mapping(inventory: Inventory) -> dict:
    """Recover drive-PCI -> /dev/unvmeN via the sysfs identity primary path.

    Falls back to an empty mapping (interim bootstrap would be used at the
    bind-site) when the driver-published sysfs attributes are absent.
    """
    mapping, _warnings = identity_mod.correlate_unvme_devices(inventory)
    by_pci: dict = {}
    for unvme_dev, drive in mapping.items():
        if drive.pci_bus_id:
            by_pci[drive.pci_bus_id] = unvme_dev
    return by_pci


def _rollback_side_effects(
    launched: List[Tuple[str, str, str]],
    release_pci: List[str],
    started_mps: bool = False,
) -> Tuple[List[str], List[str]]:
    """Best-effort teardown of one apply's side effects, in reverse order.

    Undoes setup (bind -> MPS -> launch) in reverse: stop the domains this
    invocation launched, then stop the host-global MPS daemon **iff this apply
    started it** (it was not already running -- the same "undo only what we did"
    rule as ``release_pci``; a daemon another config relies on is left alone),
    then release only the drives it newly bound.

    Teardown **halts** if any domain fails to stop: a still-running server holds
    its ``/dev/unvmeN`` open and may be using MPS, so unbinding its drives or
    quitting the daemon underneath it risks wedging the device or corrupting live
    I/O. In that case the remaining resources are deliberately left in place and
    reported, so ``stop``/``doctor`` (or an operator) can finish the cleanup once
    the process is gone.

    Returns ``(errors, still_running)`` -- the teardown-failure messages (empty
    when the rollback was clean) and the names of the domains this invocation
    launched but could NOT stop (so the caller records exactly those, and not the
    ones it did successfully stop).
    """
    errors: List[str] = []
    stop_failed: List[str] = []
    for cfg, dom, storage_backend in reversed(launched):
        if storage_backend == "lender":
            lr = launcher.stop_domain(cfg, dom, storage_backend, dry_run=False)
        else:
            lr = launcher.stop_domain(cfg, dom, dry_run=False)
        if not lr.ok:
            errors.append(f"rollback: could not stop domain {dom}: {_fail_detail(lr)}")
            stop_failed.append(dom)
    if stop_failed:
        # Leave MPS and the drive bindings alone -- see the docstring. The caller
        # (_fail_with_rollback) sees a non-empty error list and persists a residual
        # registry record so the still-running domains stay tracked.
        errors.append(
            f"rollback halted: {len(stop_failed)} domain(s) still running "
            f"({', '.join(stop_failed)}); left MPS and drive bindings intact rather "
            f"than releasing resources still in use -- resolve those processes, then "
            f"re-run 'scadactl stop' to finish the teardown"
        )
        return errors, stop_failed
    if started_mps:
        mr = mps.stop(dry_run=False)
        if not mr.ok:
            errors.append(f"rollback: could not stop MPS: {_fail_detail(mr)}")
    if release_pci:
        br = binding.unbind_drives(release_pci, dry_run=False)
        if not br.ok:
            errors.append(
                f"rollback: could not release {len(release_pci)} drive(s): {_fail_detail(br)}"
            )
    return errors, stop_failed


def _fail_with_rollback(
    result: ApplyResult,
    existing: Registry,
    config: Config,
    launched: List[Tuple[str, str, str]],
    release_pci: List[str],
    reg_domains: Optional[List[RegistryDomain]] = None,
    started_mps: bool = False,
) -> ApplyResult:
    """Handle a post-mutation apply failure via rollback.

    The originating failure is already in ``result.errors``. Roll back this
    invocation's side effects best-effort; if the rollback is clean, leave the
    registry untouched (the host is back to its prior state). If the rollback is
    incomplete, persist a registry record of the domains we believe are still
    running so ``stop``/``doctor`` can finish the cleanup, and tell the operator
    how to rectify. Returns ``result``.
    """
    teardown_errors, still_running = _rollback_side_effects(
        launched, release_pci, started_mps=started_mps
    )
    if not teardown_errors:
        result.actions.append(
            "rolled back all changes from this apply (host left unchanged)"
        )
        return result

    # Rollback incomplete: report it and persist what is (believed to be) still
    # running so it is not orphaned/invisible. Record ONLY the domains we could
    # not stop -- the ones we did stop are gone and must not be re-recorded.
    result.errors.extend(teardown_errors)
    launched_names = {name for _cfg, name, _backend in launched}
    residual = [rd for rd in (reg_domains or []) if rd.name in set(still_running)]
    if residual:
        def _publish_residual(cur):
            # reconcile_add_config REPLACES this config's entries wholesale, so the
            # residual list must also carry the config's domains this apply never
            # touched (healthy no-op/self-heal siblings). Without them a partial
            # failure would delete a live domain's registry record, orphaning a
            # running server that 'stop --config' can then no longer find. Groups
            # come from the config for the same reason -- passing {} would wipe the
            # group map that client selection depends on.
            untouched = [
                d for d in cur.domains
                if d.config == config.name and d.name not in launched_names
            ]
            return registry_mod.reconcile_add_config(
                cur,
                config.name,
                untouched + residual,
                domain_groups=config.domain_groups,
                mps_enabled=config.mps_enabled,
                mps_multiuser=config.mps_multiuser,
            )

        try:
            result.registry = registry_mod.update_registry(_publish_residual)
            result.actions.append(
                f"rollback incomplete: recorded {len(residual)} still-running "
                f"domain(s) in {registry_mod.registry_path()}; run "
                f"'scadactl stop --config {config.name}' to retry cleanup or re-run apply"
            )
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            result.errors.append(f"could not persist residual registry: {exc}")
    else:
        result.actions.append(
            "rollback incomplete and no launched domains to record; resolve "
            "manually (e.g. 'scadactl stop --all' or 'scada-unload-modules')"
        )
    return result


@_host_locked
def stop_config(config_name: str, dry_run: bool = False) -> ApplyResult:
    """Stop one config's domains and release only its drives.

    A domain is dropped from the registry only once it is *fully* torn down
    (its process stopped AND its drives released). A domain whose stop or drive
    release fails stays tracked and the failure is reported (non-zero exit), so
    ``stop`` never silently loses a still-running process; re-running retries it.
    """
    result = ApplyResult(config=config_name, dry_run=dry_run)
    existing = registry_mod.read_registry()
    owned = [d for d in existing.domains if d.config == config_name]
    if not owned:
        result.actions.append(f"config {config_name} not active; nothing to stop")
        result.registry = existing
        return result

    # Read the stopped config's MPS flag before we mutate the registry, so we can
    # tear down the host-global MPS daemon if this was the last config using it.
    stopped_mps = any(c.mps_enabled for c in existing.configs if c.name == config_name)

    removed: List[Tuple[str, str]] = []  # (config, domain) fully torn down
    for d in owned:
        # A domain whose recorded process is already gone needs no stop. This
        # keeps stop idempotent so a rerun -- e.g. after a prior MPS-teardown
        # failure, where the domains were already stopped -- can complete the
        # teardown and retry MPS cleanup, rather than failing on the now-absent
        # pidfile (stop_domain is not idempotent for an already-stopped domain).
        already_dead = (
            not dry_run and bool(d.pid) and not reconcile_mod._domain_alive(d)
        )
        if not already_dead:
            if d.storage_backend == "lender":
                lr = launcher.stop_domain(
                    config_name, d.name, d.storage_backend, dry_run=dry_run
                )
            else:
                lr = launcher.stop_domain(config_name, d.name, dry_run=dry_run)
            if not dry_run and not lr.ok:
                result.errors.append(f"domain {d.name}: stop failed: {_fail_detail(lr)}")
                continue
        # Lender consumes already-VFIO-bound BDFs and must retain their binding.
        pci = [drv.pci_bus_id for drv in d.drives if drv.pci_bus_id]
        if d.storage_backend == "kernel" and pci:
            br = binding.unbind_drives(pci, dry_run=dry_run)
            if not dry_run and not br.ok:
                result.errors.append(
                    f"domain {d.name}: drive release failed: {_fail_detail(br)}"
                )
                continue
        if not dry_run and _remove_managed_profiler_socket(d):
            result.actions.append(f"removed stale profiler socket for {d.name}")
        removed.append((config_name, d.name))

    result.actions.append(
        f"stopped {len(removed)} of {len(owned)} domain(s) for config {config_name}"
    )
    if len(removed) < len(owned):
        result.actions.append(
            f"{len(owned) - len(removed)} domain(s) still tracked (teardown "
            f"incomplete); re-run 'scadactl stop --config {config_name}' after "
            f"resolving the errors above"
        )

    # Tear down the host-global MPS daemon BEFORE dropping this config from the
    # registry, when THIS config needed it, its teardown is complete, and no OTHER
    # active config still relies on it. Doing MPS first means a failed teardown can
    # keep the owning config tracked for a retry -- otherwise the daemon leaks and a
    # rerun (or `stop --all`) sees no MPS-enabled config and never stops it.
    complete = len(removed) == len(owned)
    remaining_mps = any(
        c.mps_enabled for c in existing.configs if c.name != config_name
    )
    mps_ok = True
    if stopped_mps and complete and not remaining_mps:
        mr = mps.stop(dry_run=dry_run)
        if not dry_run and not mr.ok:
            result.errors.append(f"MPS stop failed: {_fail_detail(mr)}")
            result.actions.append(
                f"registry retained (MPS stop failed); re-run "
                f"'scadactl stop --config {config_name}' to retry MPS cleanup"
            )
            mps_ok = False
        else:
            result.actions.append("stop MPS (last MPS-enabled config stopped)")

    # Publish the registry. Drop this config's torn-down domains only once MPS
    # teardown has succeeded (or was not needed); if MPS stop failed, keep them so
    # a rerun still sees the owning config and retries the MPS cleanup.
    publish_removed = removed if mps_ok else []
    if dry_run:
        result.registry = registry_mod.reconcile_remove_domains(existing, publish_removed)
    else:
        # Locked read + reconcile + write so a concurrent writer isn't clobbered:
        # only the domains we tore down are removed from the latest on-disk state.
        result.registry = registry_mod.update_registry(
            lambda cur: registry_mod.reconcile_remove_domains(cur, publish_removed)
        )
    return result


@_host_locked
def stop_all(dry_run: bool = False) -> ApplyResult:
    """Stop all managed configs/domains and unload the module.

    Domains that fail to stop stay tracked and their failures are reported. The
    host-global teardown (module unload + MPS stop) runs only when every domain
    stopped cleanly -- unloading the module while a process still holds a
    ``/dev/unvmeN`` would fail anyway. If the module unload itself fails, the
    registry is retained (drives may still be bound) so nothing is orphaned.
    Re-running ``stop --all`` retries.
    """
    result = ApplyResult(config="(all)", dry_run=dry_run)
    existing = registry_mod.read_registry()

    failed: List[RegistryDomain] = []
    removed: List[Tuple[str, str]] = []  # (config, domain) actually stopped
    for d in existing.domains:
        # Skip stopping a domain whose recorded process is already gone (idempotent
        # rerun, e.g. after a prior MPS/unload failure) -- count it as stopped so
        # the rerun re-reaches the host-global teardown.
        already_dead = (
            not dry_run and bool(d.pid) and not reconcile_mod._domain_alive(d)
        )
        if not already_dead:
            if d.storage_backend == "lender":
                lr = launcher.stop_domain(
                    d.config, d.name, d.storage_backend, dry_run=dry_run
                )
            else:
                lr = launcher.stop_domain(d.config, d.name, dry_run=dry_run)
            if not dry_run and not lr.ok:
                result.errors.append(
                    f"domain {d.name} (config {d.config}): stop failed: {_fail_detail(lr)}"
                )
                failed.append(d)
                continue
        if not dry_run and _remove_managed_profiler_socket(d):
            result.actions.append(f"removed stale profiler socket for {d.name}")
        removed.append((d.config, d.name))
    if existing.domains:
        result.actions.append(
            f"stopped {len(existing.domains) - len(failed)} of "
            f"{len(existing.domains)} domain(s)"
        )

    if failed:
        # Some domains still running: skip the host-global unload/MPS stop
        # (unsafe while devices are in use) and keep the registry intact.
        result.errors.append(
            f"{len(failed)} domain(s) still running; skipped module unload and "
            f"MPS stop -- re-run 'scadactl stop --all' after resolving the errors above"
        )
        result.registry = existing
        return result

    # Every domain stopped: tear down host-global state.
    unload_ok = True
    mps_ok = True
    kernel_domains = any(d.storage_backend == "kernel" for d in existing.domains)
    if existing.active_configs and kernel_domains:
        br = binding.unbind_all(dry_run=dry_run)
        if not dry_run and not br.ok:
            result.errors.append(f"full unbind/module unload failed: {_fail_detail(br)}")
            unload_ok = False
        else:
            result.actions.append("full unbind + module unload")
    if existing.active_configs:
        # Only quit the host-global MPS daemon if a managed config actually
        # enabled it; a non-MPS config must not disturb an MPS daemon that may
        # be serving unrelated workloads.
        if any(c.mps_enabled for c in existing.configs):
            mr = mps.stop(dry_run=dry_run)
            if not dry_run and not mr.ok:
                result.errors.append(f"MPS stop failed: {_fail_detail(mr)}")
                mps_ok = False
            else:
                result.actions.append("stop MPS")

    if not unload_ok or not mps_ok:
        # Drives may still be bound and/or the MPS daemon may still be up; keep the
        # registry so the failed teardown stays tracked and a rerun retries it (the
        # already-dead domains are skipped on rerun, so it re-reaches this step).
        reason = "module unload failed" if not unload_ok else "MPS stop failed"
        result.actions.append(
            f"registry retained ({reason}); re-run 'scadactl stop --all'"
        )
        result.registry = existing
        return result

    # Remove only the domains we stopped (locked read + reconcile + write), so a
    # config added by a concurrent apply is preserved rather than cleared.
    if dry_run:
        result.registry = registry_mod.reconcile_remove_domains(existing, removed)
    else:
        result.registry = registry_mod.update_registry(
            lambda cur: registry_mod.reconcile_remove_domains(cur, removed)
        )
    return result
