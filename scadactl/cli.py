# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scadactl`` CLI: argparse dispatch + ``--help`` for every command.

Commands: inventory, config {init,validate,
add-domain}, apply, stop, status, logs, domains, run, env, check-client,
doctor.

The CLI is a thin dispatch layer; the real logic lives in the typed modules so
it stays unit-testable. Hardware/root-dependent commands surface actionable
errors rather than crashing when NVML/root are absent.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import List, Optional, Tuple

from scadactl import __version__
from scadactl import (
    apply as apply_mod,
    client as client_mod,
    diagnostics,
    identity as identity_mod,
    inventory as inventory_mod,
    models,
    plan as plan_mod,
    policy as policy_mod,
    config as config_mod,
    reconcile as reconcile_mod,
    registry as registry_mod,
    store as store_mod,
    wizard as wizard_mod,
)


def _eprint(*args) -> None:
    print(*args, file=sys.stderr)


def _now() -> str:
    """Human-readable local timestamp for config created/modified fields."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _stamp() -> str:
    """Filename-safe local timestamp (sortable, no colons)."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _resolve_config(args) -> Tuple[config_mod.Config, str]:
    """Resolve a config from a positional store NAME or -f FILE.

    Exactly one of the two must be given. Returns (config, source-label).
    Raises ValueError / OSError / store_mod.StoreError on problems.
    """
    name = getattr(args, "name", None)
    fpath = getattr(args, "file", None)
    if name and fpath:
        raise ValueError("specify either a config name or -f FILE, not both")
    if fpath:
        return config_mod.load_config(fpath), f"file {fpath}"
    if name:
        return store_mod.load(name), f"stored config '{name}'"
    raise ValueError("specify a config name (from the store) or -f FILE")


def _render_dry_run(prof, inv, reg, show_commands: bool, as_json: bool) -> int:
    """Shared dry-run preview for `apply --dry-run`."""
    res = config_mod.validate_config(prof, inventory=inv, registry=reg)
    plan = plan_mod.compute_plan(prof, inventory=inv, registry=reg)
    # The reconcile plan is the *delta* vs. what's actually running (no-op,
    # self-heal, reclaim, or refusal). It needs host state, so is only available
    # when inventory loaded.
    rplan = (
        reconcile_mod.plan_reconciliation(
            prof, inv, reg, apply_mod.build_registry_domains(prof, inv)
        )
        if inv is not None
        else None
    )
    ok = res.ok and (rplan is None or rplan.ok)
    if as_json:
        out = plan.to_dict()
        out["validation"] = {
            "ok": res.ok,
            "errors": res.errors,
            "warnings": res.warnings,
        }
        if rplan is not None:
            out["reconcile"] = {
                "ok": rplan.ok,
                "errors": rplan.errors,
                "actionable": [dp.name for dp in rplan.actionable()],
                "reclaim_pci": rplan.reclaim_pci(),
                "domains": [
                    {"name": dp.name, "action": dp.action.value, "reasons": dp.reasons}
                    for dp in rplan.domains
                ],
            }
        if show_commands:
            out["commands"] = apply_mod.planned_commands(prof, inv)
        print(json.dumps(out, indent=2))
        return 0 if ok else 1
    print("DRY RUN — no host changes will be made.\n")
    for w in res.warnings:
        print(f"warning: {w}")
    for e in res.errors:
        print(f"error: {e}")
    if res.errors:
        print("(apply would refuse the errors above)\n")
    print(plan_mod.render_plan(plan))
    if rplan is not None:
        print()
        print(_render_reconcile(rplan))
    if show_commands:
        print("\nCommands:")
        for c in apply_mod.planned_commands(prof, inv):
            print(f"  $ {c}")
    else:
        print("\n(use --commands to see the exact commands; remove --dry-run to execute)")
    return 0 if ok else 1


def _render_reconcile(rplan) -> str:
    """Human-readable reconcile delta for the dry-run preview."""
    lines = ["Reconcile (delta vs. what is running):"]
    if not rplan.ok:
        for e in rplan.errors:
            lines.append(f"  error: {e}")
        lines.append("  (apply would refuse — resolve the above first)")
        return "\n".join(lines)
    if not rplan.actionable():
        lines.append("  no changes: everything requested is already running (no-op)")
        return "\n".join(lines)
    for dp in rplan.actionable():
        lines.append(f"  {dp.name}: {dp.action.value} — {'; '.join(dp.reasons)}")
    if rplan.reclaim_pci():
        lines.append(
            "  WARNING: will tear down and rebuild unmanaged resources currently set "
            f"up on the host — drive(s) {', '.join(rplan.reclaim_pci())}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #
def cmd_inventory(args) -> int:
    inv = inventory_mod.collect_inventory()
    if args.json:
        print(json.dumps(inv.to_dict(), indent=2))
    else:
        print(inventory_mod.render_inventory(inv))
    return 0


def _load_inventory_best_effort():
    try:
        return inventory_mod.collect_inventory()
    except Exception as exc:  # noqa: BLE001
        _eprint(f"warning: inventory unavailable ({exc}); identity checks skipped")
        return None


def cmd_config_init(args) -> int:
    name = args.name or "scadactl"
    if args.empty:
        prof = config_mod.empty_config(name)
    elif args.gpus or args.drives:
        inv = inventory_mod.collect_inventory()
        try:
            prof = config_mod.seed_config_from_resources(
                inv,
                name,
                _split(args.gpus),
                _split(args.drives),
            )
        except ValueError as exc:
            _eprint(f"error: {exc}")
            return 2
    else:
        prof = wizard_mod.interactive_init(
            name, inventory_mod.collect_inventory(), force=args.force
        )
        if prof is None:
            return 1

    # Stamp created/modified.
    prof.modified = _now()
    prof.created = prof.created or prof.modified

    # Primary outcome: save into the managed store (needs root on a default
    # install). Fall back to a file copy so work is never lost.
    stored = False
    try:
        path = store_mod.save(prof, overwrite=args.force)
        print(f"saved config '{prof.name}' to the store: {path}")
        stored = True
    except store_mod.StoreError as exc:
        _eprint(f"warning: could not save to the config store: {exc}")

    out = args.output
    if not stored and not out:
        out = f"{os.path.basename(prof.name)}-{_stamp()}.yaml"
        _eprint(
            f"writing a local copy to {out}; an admin can register it later with "
            f"'scadactl config import {out}'."
        )
    if out:
        config_mod.save_config(prof, out)
        print(f"wrote {os.path.abspath(out)}")
    return 0


def _config_ref(args) -> str:
    """Render how the user referenced the config, for next-step hints."""
    return f"-f {args.file}" if getattr(args, "file", None) else (args.name or "")


def cmd_config_validate(args) -> int:
    try:
        prof, _src = _resolve_config(args)
    except (OSError, ValueError, store_mod.StoreError) as exc:
        _eprint(f"error: {exc}")
        return 2
    inv = _load_inventory_best_effort()
    reg = registry_mod.read_registry()
    res = config_mod.validate_config(prof, inventory=inv, registry=reg)
    if args.json:
        print(json.dumps({"ok": res.ok, "errors": res.errors, "warnings": res.warnings}, indent=2))
    else:
        for w in res.warnings:
            print(f"warning: {w}")
        for e in res.errors:
            print(f"error: {e}")
        if res.ok:
            ndrives = sum(len(d.drives) for d in prof.domains)
            print(
                f"OK — config '{prof.name}' is valid "
                f"({len(prof.domains)} domains, {ndrives} drives)."
            )
            print(f"Next: scadactl apply {_config_ref(args)} --dry-run   (preview)")
        else:
            print("INVALID")
    return 0 if res.ok else 1


def cmd_config_list(args) -> int:
    try:
        names = store_mod.list_names()
    except store_mod.StoreError as exc:
        _eprint(f"error: {exc}")
        return 2
    reg = registry_mod.read_registry()
    active = set(reg.active_configs)
    rows = []
    unreadable: List[Tuple[str, str]] = []
    for n in names:
        try:
            p = store_mod.load(n)
        except store_mod.StoreError as exc:
            # A config that exists but cannot be read must NOT be silently dropped:
            # that reads as "no configs" while `config init` still reports it exists
            # (a bare stat needs only directory traverse, reading needs the file be
            # world-readable). Surface it so the mismatch is diagnosable.
            unreadable.append((n, str(exc)))
            continue
        rows.append(
            {
                "name": p.name,
                "domains": len(p.domains),
                "drives": sum(len(d.drives) for d in p.domains),
                "mps": p.mps_enabled,
                "mps_multiuser": p.mps_multiuser,
                "created": p.created,
                "modified": p.modified,
                "active": p.name in active,
            }
        )
    for n, err in unreadable:
        _eprint(f"warning: config '{n}' is present but could not be read: {err}")
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        if unreadable:
            print(
                f"{len(unreadable)} config(s) present in {store_mod.config_dir()} "
                f"but unreadable (see warnings above)"
            )
        else:
            print(f"no stored configs in {store_mod.config_dir()}")
        return 0
    print(f"{'name':20} {'domains':7} {'drives':6} {'mps':4} {'active':6} modified")
    for r in rows:
        print(
            f"{r['name']:20} {r['domains']:<7} {r['drives']:<6} "
            f"{('on' if r['mps'] else 'off'):4} "
            f"{('yes' if r['active'] else '-'):6} {r['modified'] or ''}"
        )
    return 0


def cmd_config_show(args) -> int:
    try:
        prof, _src = _resolve_config(args)
    except (OSError, ValueError, store_mod.StoreError) as exc:
        _eprint(f"error: {exc}")
        return 2
    if args.json:
        print(json.dumps(prof.to_dict(), indent=2))
    else:
        print(config_mod.summarize(prof, _load_inventory_best_effort()))
    return 0


def cmd_config_edit(args) -> int:
    try:
        prof, _src = _resolve_config(args)
    except (OSError, ValueError, store_mod.StoreError) as exc:
        _eprint(f"error: {exc}")
        return 2
    if not sys.stdin.isatty():
        _eprint("error: 'config edit' requires an interactive terminal")
        return 2
    inv = inventory_mod.collect_inventory()
    if not inv.gpus:
        for w in inv.warnings:
            _eprint(f"warning: {w}")
        _eprint("error: no GPUs detected; cannot edit interactively")
        return 1
    try:
        edited = wizard_mod.ConfigWizard(inv, prof.name, config=prof).run(
            start=wizard_mod.ConfigWizard.SUMMARY
        )
    except KeyboardInterrupt:
        _eprint("\naborted; no changes written.")
        return 1
    if edited is None:
        return 1
    edited.modified = _now()
    edited.created = edited.created or _now()
    if args.file:
        config_mod.save_config(edited, args.file)
        print(f"wrote {os.path.abspath(args.file)}")
        return 0
    try:
        store_mod.save(edited, overwrite=True)
    except store_mod.StoreError as exc:
        _eprint(f"error: {exc}")
        return 2
    print(f"updated '{edited.name}' in the config store")
    return 0


def cmd_config_rm(args) -> int:
    try:
        store_mod.delete(args.name)
    except store_mod.StoreError as exc:
        _eprint(f"error: {exc}")
        return 2
    print(f"removed '{args.name}' from the config store")
    return 0


def cmd_config_import(args) -> int:
    try:
        prof = config_mod.load_config(args.file)
    except (OSError, ValueError) as exc:
        _eprint(f"error: {exc}")
        return 2
    prof.modified = _now()
    prof.created = prof.created or prof.modified
    try:
        path = store_mod.save(prof, overwrite=args.force)
    except store_mod.StoreError as exc:
        _eprint(f"error: {exc}")
        return 2
    print(f"imported '{prof.name}' -> {path}")
    return 0


def cmd_config_compat(args) -> int:
    refs: List[Tuple[str, str]] = [("name", n) for n in (args.names or [])]
    refs += [("file", f) for f in (args.file or [])]
    if len(refs) < 2:
        _eprint("error: give at least two configs (names and/or -f FILE) to compare")
        return 2
    configs = []
    labels = []
    for kind, ref in refs:
        try:
            p = config_mod.load_config(ref) if kind == "file" else store_mod.load(ref)
        except (OSError, ValueError, store_mod.StoreError) as exc:
            _eprint(f"error: {exc}")
            return 2
        configs.append(p)
        labels.append(ref if kind == "name" else f"-f {ref}")
    res = config_mod.validate_composition(configs)
    mps_on = [p.name for p in configs if p.mps_enabled]
    if args.json:
        print(json.dumps({
            "configs": labels,
            "compatible": res.ok,
            "conflicts": res.errors,
            "shared_mps": mps_on if len(mps_on) > 1 else [],
        }, indent=2))
        return 0 if res.ok else 1
    print(f"Configs: {', '.join(labels)}")
    if res.ok:
        print("✓ compatible — no GPU, drive, or domain-name overlap.")
    else:
        print("✗ conflict — cannot run concurrently:")
        for e in res.errors:
            print(f"  - {e}")
    if len(mps_on) > 1 and res.ok:
        print(f"note: MPS is host-global and enabled by {', '.join(mps_on)} (shared, not a conflict).")
    return 0 if res.ok else 1


def cmd_config_add_domain(args) -> int:
    inv = inventory_mod.collect_inventory()
    # -f names both the config to extend and the output path. Extend it only if
    # it already exists; otherwise start a fresh config so add-domain can create
    # one (the file is written at the end via --output / -f).
    if args.file and os.path.exists(args.file):
        try:
            prof = config_mod.load_config(args.file)
        except (OSError, ValueError) as exc:
            _eprint(f"error: {exc}")
            return 2
    else:
        prof = config_mod.empty_config(args.config_name or "scadactl")
    server_env = {}
    for kv in args.env or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            server_env[k] = v
    try:
        config_mod.add_domain(
            prof,
            inv,
            name=args.name,
            gpu_token=args.gpu,
            drive_tokens=_split(args.drives),
            server_env=server_env,
            socket_group=args.socket_group,
        )
    except ValueError as exc:
        _eprint(f"error: {exc}")
        return 2
    out = args.output or args.file
    if out:
        config_mod.save_config(prof, out)
        print(f"wrote {out}")
    else:
        print(config_mod.dump_config(prof))
    return 0


def cmd_apply(args) -> int:
    # A real apply mutates the host (binds drives, starts MPS/domains, writes the
    # registry) and takes the root-only host lock, so gate it up front -- BEFORE
    # reading the config -- so a non-root user gets "run as root", not an incidental
    # config-read or host.lock PermissionError. --dry-run is a read-only preview and
    # stays unprivileged.
    if not args.dry_run:
        rc = _require_root(
            "scadactl apply",
            "it binds drives, starts services, and writes the registry",
        )
        if rc is not None:
            return rc
    try:
        prof, _src = _resolve_config(args)
    except (OSError, ValueError, store_mod.StoreError) as exc:
        _eprint(f"error: {exc}")
        return 2
    if args.dry_run:
        inv = _load_inventory_best_effort()
        reg = registry_mod.read_registry()
        return _render_dry_run(prof, inv, reg, show_commands=args.commands, as_json=args.json)
    inv = inventory_mod.collect_inventory()
    res = apply_mod.apply_config(prof, inv, dry_run=False, force=args.force)
    for a in res.actions:
        print(a)
    for e in res.errors:
        _eprint(f"error: {e}")
    return 0 if res.ok else 1


def cmd_stop(args) -> int:
    if args.all and args.config:
        _eprint("error: --all and --config are mutually exclusive")
        return 2
    if not args.all and not args.config:
        _eprint("error: specify --config NAME or --all")
        return 2
    # A real stop mutates the host (stops domains, releases drives, writes the
    # registry) under the root-only host lock; gate it up front. --dry-run previews
    # only and stays unprivileged.
    if not args.dry_run:
        rc = _require_root(
            "scadactl stop",
            "it stops domains, releases drives, and writes the registry",
        )
        if rc is not None:
            return rc
    if args.all:
        res = apply_mod.stop_all(dry_run=args.dry_run)
    else:  # args.config, validated above
        res = apply_mod.stop_config(args.config, dry_run=args.dry_run)
    for a in res.actions:
        print(a)
    for e in res.errors:
        _eprint(f"error: {e}")
    return 0 if res.ok else 1


def cmd_status(args) -> int:
    reg = registry_mod.read_registry()
    if args.json:
        print(json.dumps(reg.to_dict(), indent=2))
    else:
        print(diagnostics.render_status(reg, details=args.details))
    return 0


def cmd_logs(args) -> int:
    reg = registry_mod.read_registry()
    logs = diagnostics.collect_logs(reg, _split(args.domains) if args.domains else None)
    if args.json:
        print(json.dumps(logs, indent=2))
    else:
        for dom, files in logs.items():
            print(f"{dom}:")
            for f in files:
                print(f"  {f}")
            if not files:
                print("  (no logs found)")
    return 0


def cmd_domains(args) -> int:
    reg = registry_mod.read_registry()
    views = client_mod.list_domains(reg)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": v.name,
                        "host_gpu": v.host_gpu,
                        "gpu_uuid": v.gpu_uuid,
                        "container_cuda": v.container_cuda,
                        "socket": v.socket,
                        "status": v.status,
                    }
                    for v in views
                ],
                indent=2,
            )
        )
    else:
        print(f"{'domain':14} {'host_gpu':9} {'gpu_uuid':12} {'container_cuda':14} {'socket':40} status")
        for v in views:
            cc = "-" if v.container_cuda is None else str(v.container_cuda)
            print(
                f"{v.name:14} {(v.host_gpu or ''):9} {v.gpu_uuid:12} {cc:14} "
                f"{v.socket:40} {v.status}"
            )
    return 0


def cmd_run(args) -> int:
    reg = registry_mod.read_registry()
    selection = _require_selection(args)
    if not selection:
        return 2
    command = _normalize_command(args.argv)
    if not command:
        _eprint("error: no command given after --")
        return 2
    try:
        return client_mod.run_command(reg, selection, command)
    except (KeyError, client_mod.GpuNotVisible, RuntimeError) as exc:
        return _client_error(exc)
    except OSError as exc:
        _eprint(f"error: cannot run command: {exc}")
        return 127


def cmd_env(args) -> int:
    reg = registry_mod.read_registry()
    selection = _require_selection(args)
    if not selection:
        return 2
    try:
        domains = client_mod.resolve_selection(reg, selection)
        if args.json:
            print(json.dumps(client_mod.env_json(reg, domains), indent=2))
        else:
            env, _ = client_mod.compute_env(reg, domains)
            print(client_mod.render_env_exports(env))
    except (KeyError, client_mod.GpuNotVisible, RuntimeError) as exc:
        return _client_error(exc)
    return 0


def cmd_check_client(args) -> int:
    reg = registry_mod.read_registry()
    if not args.domain:
        _eprint("error: specify --domain NAME")
        return 2
    ok, msg = client_mod.check_client(reg, args.domain)
    print(msg)
    return 0 if ok else 1


def cmd_doctor(args) -> int:
    reg = registry_mod.read_registry()
    inv = _load_inventory_best_effort()
    report = diagnostics.run_doctor(reg, inventory=inv)
    print(diagnostics.render_doctor(report))
    if args.logs:
        print("\nLogs:")
        logs = diagnostics.collect_logs(reg)
        for dom, files in logs.items():
            print(f"  {dom}: {', '.join(files) or '(none)'}")
    return 0 if report.ok else 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _split(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _require_selection(args) -> Optional[str]:
    """Return the --domain/--domains selection, or print an error and None."""
    selection = args.domain or args.domains
    if not selection:
        _eprint("error: specify --domain NAME or --domains a,b")
    return selection


def _normalize_command(argv: Optional[List[str]]) -> List[str]:
    """Drop the leading ``--`` that argparse.REMAINDER keeps before the command."""
    argv = list(argv or [])
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def _client_error(exc: Exception) -> int:
    """Map a client-plane exception to a printed message and exit code."""
    if isinstance(exc, KeyError):
        _eprint(f"error: unknown domain or group {exc}")
        return 2
    if isinstance(exc, client_mod.GpuNotVisible):
        _eprint(f"error: {exc}")
        return 3
    _eprint(f"error: {exc}")  # RuntimeError (e.g. NVML unavailable)
    return 4


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Device policy (host allow/block lists) -- admin only
# --------------------------------------------------------------------------- #
def _require_root(action: str, why: str = "it mutates host state") -> Optional[int]:
    """Return an error code if not root; host-mutating commands are root-only.

    ``action`` is the command (e.g. ``"scadactl apply"``); ``why`` a short reason.
    Checked BEFORE the command reads configs or touches the host, so an
    unprivileged invocation fails fast with a clear message rather than a deep,
    incidental ``PermissionError`` (e.g. on the config store or the host lock).
    """
    if os.geteuid() != 0:
        _eprint(f"error: {action} must be run as root ({why}); try: sudo {action} …")
        return 2
    return None


def _split_tokens(csv: Optional[str]) -> List[str]:
    return [t for t in (csv or "").replace(",", " ").split() if t]


def _resolve_policy_devices(inv, gpu_tokens, drive_tokens):
    """Resolve --gpus/--drives tokens to (gpus, drives) or raise ValueError."""
    gpus, drives, unknown = [], [], []
    for t in gpu_tokens:
        g = identity_mod.resolve_gpu(inv, t)
        (gpus if g is not None else unknown).append(g if g is not None else t)
    for t in drive_tokens:
        d = identity_mod.resolve_drive(inv, t)
        (drives if d is not None else unknown).append(d if d is not None else t)
    if unknown:
        raise ValueError(
            f"unknown device(s): {', '.join(unknown)} (see 'scadactl inventory')"
        )
    return gpus, drives


def _do_policy_edit(args, list_kind: str, add: bool, verb: str) -> int:
    if (rc := _require_root(f"scadactl policy {verb}",
                            "it changes host device policy")) is not None:
        return rc
    gpu_tokens = _split_tokens(getattr(args, "gpus", None))
    drive_tokens = _split_tokens(getattr(args, "drives", None))
    if not gpu_tokens and not drive_tokens:
        _eprint("error: specify --gpus and/or --drives")
        return 2
    inv = inventory_mod.collect_inventory()
    try:
        gpus, drives = _resolve_policy_devices(inv, gpu_tokens, drive_tokens)
    except ValueError as exc:
        _eprint(f"error: {exc}")
        return 2

    pol = policy_mod.load()
    reason = getattr(args, "reason", None)
    stamp = _now()

    def edit(entries, key, pci):
        if add:
            policy_mod.upsert(
                entries, models.PolicyEntry(key=key, pci_bus_id=pci,
                                            reason=reason, added=stamp)
            )
        else:
            policy_mod.remove(entries, key, pci)

    for g in gpus:
        edit(pol.gpu_block if list_kind == "block" else pol.gpu_allow,
             g.uuid, g.pci_bus_id)
    for d in drives:
        edit(pol.drive_block if list_kind == "block" else pol.drive_allow,
             d.serial, d.pci_bus_id)

    try:
        path = policy_mod.save(pol)
    except policy_mod.PolicyError as exc:
        _eprint(f"error: {exc}")
        return 2

    verbed = {("block", True): "blocked", ("block", False): "unblocked",
              ("allow", True): "added to allow-list",
              ("allow", False): "removed from allow-list"}[(list_kind, add)]
    names = [g.alias for g in gpus] + [d.alias for d in drives]
    print(f"{verbed}: {', '.join(names)}  ({path})")
    return 0


def cmd_policy_block(args) -> int:
    return _do_policy_edit(args, "block", add=True, verb="block")


def cmd_policy_unblock(args) -> int:
    return _do_policy_edit(args, "block", add=False, verb="unblock")


def cmd_policy_allow(args) -> int:
    return _do_policy_edit(args, "allow", add=True, verb="allow")


def cmd_policy_unallow(args) -> int:
    return _do_policy_edit(args, "allow", add=False, verb="unallow")


def _policy_alias(inv, entry, is_gpu: bool) -> str:
    if inv is None:  # inventory unavailable -> degrade to the raw key/PCI id
        return ""
    resolve = identity_mod.resolve_gpu if is_gpu else identity_mod.resolve_drive
    dev = resolve(inv, entry.key) or resolve(inv, entry.pci_bus_id or "")
    return dev.alias if dev is not None else ""


def _render_policy_section(title, entries, inv, is_gpu: bool) -> None:
    if not entries:
        return
    print(f"\n{title}:")
    for e in entries:
        alias = _policy_alias(inv, e, is_gpu)
        who = f"{alias:8}" if alias else f"{'':8}"
        reason = f"  reason: {e.reason}" if e.reason else ""
        added = f"  (added {e.added})" if e.added else ""
        print(f"  {who} {e.key or '':16} {e.pci_bus_id or '':16}{reason}{added}")


def cmd_policy_list(args) -> int:
    pol = policy_mod.load()
    if getattr(args, "json", False):
        print(json.dumps(pol.to_dict(), indent=2))
        return 0
    if pol.is_empty():
        print("No device policy set (all devices usable).")
        return 0
    inv = _load_inventory_best_effort()
    print(f"Device policy ({policy_mod.policy_file()})")
    _render_policy_section("GPUs blocked", pol.gpu_block, inv, is_gpu=True)
    _render_policy_section("GPUs allow-list", pol.gpu_allow, inv, is_gpu=True)
    _render_policy_section("Drives blocked", pol.drive_block, inv, is_gpu=False)
    _render_policy_section("Drives allow-list", pol.drive_allow, inv, is_gpu=False)
    return 0


def cmd_policy_clear(args) -> int:
    if (rc := _require_root("scadactl policy clear",
                            "it changes host device policy")) is not None:
        return rc
    pol = policy_mod.load()
    cleared = []
    if args.all or args.block:
        pol.gpu_block, pol.drive_block = [], []
        cleared.append("block lists")
    if args.all or args.allow:
        pol.gpu_allow, pol.drive_allow = [], []
        cleared.append("allow lists")
    if not cleared:
        _eprint("error: specify --all, --block, or --allow")
        return 2
    try:
        path = policy_mod.save(pol)
    except policy_mod.PolicyError as exc:
        _eprint(f"error: {exc}")
        return 2
    print(f"cleared {', '.join(cleared)}  ({path})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scadactl", description="SCADA orchestration CLI")
    p.add_argument("--version", action="version", version=f"scadactl {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # inventory
    sp = sub.add_parser("inventory", help="Show GPUs, drives, aliases, topology, binding state")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_inventory)

    # config
    pp = sub.add_parser("config", help="Author and validate configs")
    psub = pp.add_subparsers(dest="config_command", metavar="SUBCOMMAND")

    pinit = psub.add_parser("init", help="Walk through creating an editable config")
    pinit.add_argument("--output", "-o", help="also write a copy of the config to this path")
    pinit.add_argument("--name", help="config name")
    pinit.add_argument("--empty", action="store_true", help="write a commented schema with no domains")
    pinit.add_argument("--gpus", help="seed: comma-separated GPU aliases/uuids/pci")
    pinit.add_argument("--drives", help="seed: comma-separated drive aliases/serials/pci")
    pinit.add_argument("--force", action="store_true", help="overwrite an existing stored config of the same name")
    pinit.set_defaults(func=cmd_config_init)

    pval = psub.add_parser("validate", help="Validate a config without changing the host")
    pval.add_argument("name", nargs="?", help="stored config name")
    pval.add_argument("-f", "--file", help="config YAML file (instead of a stored name)")
    pval.add_argument("--json", action="store_true")
    pval.set_defaults(func=cmd_config_validate)

    plist = psub.add_parser("list", help="List stored configs")
    plist.add_argument("--json", action="store_true")
    plist.set_defaults(func=cmd_config_list)

    pshow = psub.add_parser("show", help="Show a stored config's details")
    pshow.add_argument("name", nargs="?", help="stored config name")
    pshow.add_argument("-f", "--file", help="config YAML file (instead of a stored name)")
    pshow.add_argument("--json", action="store_true")
    pshow.set_defaults(func=cmd_config_show)

    pedit = psub.add_parser("edit", help="Interactively edit a config")
    pedit.add_argument("name", nargs="?", help="stored config name")
    pedit.add_argument("-f", "--file", help="config YAML file (edited in place)")
    pedit.set_defaults(func=cmd_config_edit)

    prm = psub.add_parser("rm", help="Delete a stored config")
    prm.add_argument("name", help="stored config name")
    prm.set_defaults(func=cmd_config_rm)

    pimp = psub.add_parser("import", help="Register an external config file into the store")
    pimp.add_argument("file", help="config YAML file to import")
    pimp.add_argument("--force", action="store_true", help="overwrite an existing stored config of the same name")
    pimp.set_defaults(func=cmd_config_import)

    pcompat = psub.add_parser("compat", help="Check whether configs can run concurrently")
    pcompat.add_argument("names", nargs="*", help="stored config names")
    pcompat.add_argument("-f", "--file", action="append", help="config YAML file (repeatable)")
    pcompat.add_argument("--json", action="store_true")
    pcompat.set_defaults(func=cmd_config_compat)

    padd = psub.add_parser("add-domain", help="Add a domain (accepts aliases, writes stable ids)")
    padd.add_argument("--name", required=True, help="domain name")
    padd.add_argument("--gpu", required=True, help="GPU alias/uuid/pci")
    padd.add_argument("--drives", required=True, help="comma-separated drive aliases/serials/pci")
    padd.add_argument("--env", action="append", help="per-domain server env KEY=VALUE (repeatable)")
    padd.add_argument("--socket-group", help="socket group")
    padd.add_argument("--file", "-f", help="existing config to extend")
    padd.add_argument("--config-name", help="name when creating a new config")
    padd.add_argument("--output", "-o", help="output path (defaults to --file)")
    padd.set_defaults(func=cmd_config_add_domain)

    # policy: host device allow/block lists (admin)
    pol = sub.add_parser("policy", help="Mark GPUs/drives allowed or off-limits (admin)")
    polsub = pol.add_subparsers(dest="policy_command", metavar="SUBCOMMAND")

    def _dev_args(sub_p):
        sub_p.add_argument("--gpus", help="comma-separated GPU aliases/UUIDs/PCI ids")
        sub_p.add_argument("--drives", help="comma-separated drive aliases/serials/PCI ids")

    pblock = polsub.add_parser("block", help="Mark devices off-limits (block list)")
    _dev_args(pblock)
    pblock.add_argument("--reason", help="why it is off-limits (shown in listings)")
    pblock.set_defaults(func=cmd_policy_block)

    punblock = polsub.add_parser("unblock", help="Remove devices from the block list")
    _dev_args(punblock)
    punblock.set_defaults(func=cmd_policy_unblock)

    pallow = polsub.add_parser("allow", help="Confine SCADA to devices on the allow list")
    _dev_args(pallow)
    pallow.add_argument("--reason", help="note (shown in listings)")
    pallow.set_defaults(func=cmd_policy_allow)

    punallow = polsub.add_parser("unallow", help="Remove devices from the allow list")
    _dev_args(punallow)
    punallow.set_defaults(func=cmd_policy_unallow)

    plist = polsub.add_parser("list", help="Show the effective device policy")
    plist.add_argument("--json", action="store_true", help="machine-readable output")
    plist.set_defaults(func=cmd_policy_list)

    pclear = polsub.add_parser("clear", help="Clear policy lists")
    pclear.add_argument("--all", action="store_true", help="clear all lists")
    pclear.add_argument("--block", action="store_true", help="clear block lists")
    pclear.add_argument("--allow", action="store_true", help="clear allow lists")
    pclear.set_defaults(func=cmd_policy_clear)

    # apply
    sp = sub.add_parser("apply", help="Reconcile the host to the config")
    sp.add_argument("name", nargs="?", help="stored config name")
    sp.add_argument("-f", "--file", help="config YAML file (instead of a stored name)")
    sp.add_argument("--dry-run", action="store_true", help="preview without touching the host")
    sp.add_argument("--commands", action="store_true", help="(with --dry-run) print the exact commands")
    sp.add_argument("--json", action="store_true", help="(with --dry-run) machine-readable output")
    sp.add_argument("--force", action="store_true", help="override device policy (use blocked devices)")
    sp.set_defaults(func=cmd_apply)

    # stop
    sp = sub.add_parser("stop", help="Stop managed domains")
    grp = sp.add_mutually_exclusive_group()
    grp.add_argument("--config", help="stop a single config's domains")
    grp.add_argument("--all", action="store_true", help="stop all managed configs/domains")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_stop)

    # status
    sp = sub.add_parser("status", help="Show active domains, sockets, pids, health, drives")
    sp.add_argument("--details", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_status)

    # logs
    sp = sub.add_parser("logs", help="Collect server/MPS/profiler logs for selected domains")
    sp.add_argument("--domains", help="comma-separated domain names")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_logs)

    # domains
    sp = sub.add_parser("domains", help="Show active domains usable from the current environment")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_domains)

    # run
    sp = sub.add_parser("run", help="Run a command with process-local CUDA and SCADA selection")
    sp.add_argument("--domain", help="single domain name")
    sp.add_argument("--domains", help="comma-separated domain names (order preserved)")
    sp.add_argument("argv", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS...]")
    sp.set_defaults(func=cmd_run)

    # env
    sp = sub.add_parser("env", help="Print exports or JSON for manual launch")
    sp.add_argument("--domain", help="single domain name")
    sp.add_argument("--domains", help="comma-separated domain names")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_env)

    # check-client
    sp = sub.add_parser("check-client", help="Validate container visibility and domain selection")
    sp.add_argument("--domain", help="domain name to check")
    sp.set_defaults(func=cmd_check_client)

    # doctor
    sp = sub.add_parser("doctor", help="Diagnose host setup, stale sockets, missing daemons")
    sp.add_argument("--logs", action="store_true", help="also collect per-domain logs")
    sp.set_defaults(func=cmd_doctor)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    if args.command == "config" and not getattr(args, "config_command", None):
        # Print the config sub-help.
        parser.parse_args(["config", "--help"])
        return 0
    if args.command == "policy" and not getattr(args, "policy_command", None):
        parser.parse_args(["policy", "--help"])
        return 0

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
