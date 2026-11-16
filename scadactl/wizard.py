# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive config authoring for ``scadactl config init`` / ``config edit``.

Modeled as an explicit state machine (:class:`ConfigWizard`) so the
build → settings → summary/confirm → edit flow is easy to follow and test.
Inventory is dependency-injected by the caller so this module never touches
hardware directly.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from scadactl import (
    identity as identity_mod,
    config as config_mod,
    registry as registry_mod,
    store as store_mod,
)
from scadactl.models import (
    Config,
    Inventory,
    PROFILER_ENDPOINT_MODE_LEGACY_NSYS,
    PROFILER_ENDPOINT_MODE_MANAGED,
)


def _eprint(*args) -> None:
    print(*args, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Prompt helpers
# --------------------------------------------------------------------------- #
def _prompt(text: str, default: Optional[str] = None) -> str:
    """Prompt for a line of input; return ``default`` on empty/EOF."""
    label = f"{text} [{default}]: " if default not in (None, "") else f"{text}: "
    try:
        resp = input(label).strip()
    except EOFError:
        resp = ""
    return resp or (default or "")


def _prompt_yes_no(text: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            resp = input(f"{text} {suffix}: ").strip().lower()
        except EOFError:
            return default
        if not resp:
            return default
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("  please answer y or n")


def _fmt_env(env: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in env.items())


def _prompt_server_env(defaults: dict) -> dict:
    return {k: _prompt(f"  {k}", str(v)) for k, v in defaults.items()}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _store_has(name: str) -> bool:
    """Best-effort 'does a stored config with this name already exist?'.

    Never let a store read (permission, odd name) block authoring -- on any error
    just report 'no collision' and let the save path handle it.
    """
    try:
        return store_mod.exists(name)
    except Exception:  # noqa: BLE001 - store issues must not abort authoring
        return False


def _prompt_config_name(name: str, force: bool) -> str:
    """Prompt for the config name, warning + re-prompting on a store collision.

    A name that already exists in the store cannot be saved without ``--force``
    (it would otherwise fall back to a local copy), so surface the clash at name
    entry -- when the user can just rename -- instead of only at save time. When
    ``--force`` was passed the user has already opted to overwrite, so skip it.
    """
    name = _prompt("Config name", name)
    while not force and _store_has(name):
        _eprint(
            f"warning: a config named '{name}' already exists in the store; saving "
            f"would need --force to overwrite it (otherwise a local copy is written)."
        )
        if not _prompt_yes_no("Pick a different name?", default=True):
            _eprint(
                f"keeping '{name}'; re-run with --force to overwrite the existing "
                f"config, or a local copy will be written on save."
            )
            break
        name = _prompt("New config name", name)
    return name


def interactive_init(name: str, inv: Inventory, force: bool = False) -> Optional[Config]:
    """Interactively author a config from detected ``inv``.

    Returns a :class:`Config`, or ``None`` when there is nothing to write
    (non-tty, no GPUs, or the user aborted with Ctrl-C). ``force`` mirrors the
    CLI ``--force`` flag: when set, an existing-name clash is expected (overwrite)
    so the name-collision prompt is skipped.
    """
    if not sys.stdin.isatty():
        _eprint(
            "note: stdin is not a tty; emitting an empty config skeleton. "
            "Use --gpus/--drives or --empty for non-interactive seeding."
        )
        return config_mod.empty_config(name)

    if not inv.gpus:
        # Surface why the inventory is empty (e.g. missing NVML bindings, driver
        # not loaded) so this isn't a bare, mysterious "no GPUs" on a host where
        # nvidia-smi clearly shows GPUs.
        for w in inv.warnings:
            _eprint(f"warning: {w}")
        _eprint("error: no GPUs detected; cannot author a config interactively.")
        return None

    name = _prompt_config_name(name, force)
    try:
        return ConfigWizard(inv, name).run()
    except KeyboardInterrupt:
        _eprint("\naborted; not writing.")
        return None


class ConfigWizard:
    """Interactive config authoring modeled as an explicit state machine.

    State transition table (handlers return the next state)::

        BUILD_DOMAINS --(>=1 domain, "done")--> ALL_GROUP
        ALL_GROUP   --> MPS         (initial pass)  | SUMMARY (editing)
        MPS         --> PROFILER    (initial pass)  | SUMMARY (editing)
        PROFILER    --> SERVER_ENV  (initial pass)  | SUMMARY (editing)
        SERVER_ENV  --> SUMMARY
        SUMMARY     --(confirm)----------------> DONE
        SUMMARY     --(reject)-----------------> EDIT_MENU
        EDIT_MENU   --(1)----------------------> EDIT_DOMAINS
        EDIT_MENU   --(2)----------------------> ALL_GROUP
        EDIT_MENU   --(3)----------------------> MPS
        EDIT_MENU   --(4)----------------------> SERVER_ENV
        EDIT_DOMAINS --(add / modify / remove)--> SUMMARY

    The ALL_GROUP / MPS / PROFILER / SERVER_ENV handlers branch on ``self.editing``: during
    the initial pass they advance to the next linear state; once the summary has
    been shown they return to SUMMARY. ``run()`` exits on DONE (returns the
    config) or ABORT (returns ``None``). Ctrl-C raises out to the caller.

    Conventions baked in: one domain per GPU, auto-named ``<gpu_alias>_domain``;
    configs persist stable identifiers (UUID / serial); per-domain server_env
    overrides are intentionally not offered here (edit the YAML or use
    ``config add-domain`` for those).
    """

    BUILD_DOMAINS = "build_domains"
    ALL_GROUP = "all_group"
    MPS = "mps"
    PROFILER = "profiler"
    SERVER_ENV = "server_env"
    SUMMARY = "summary"
    EDIT_MENU = "edit_menu"
    EDIT_DOMAINS = "edit_domains"
    DONE = "done"
    ABORT = "abort"

    _LINEAR_NEXT = {
        BUILD_DOMAINS: ALL_GROUP,
        ALL_GROUP: MPS,
        MPS: PROFILER,
        PROFILER: SERVER_ENV,
        SERVER_ENV: SUMMARY,
    }

    def __init__(self, inv, name: str, config=None):
        self.inv = inv
        if config is not None:
            # Editing an existing config.
            self.config = config
        else:
            self.config = config_mod.empty_config(name)
            self.config.mps_enabled = True  # interactive default: MPS on
            self.config.mps_multiuser = True
        self.editing = False

    # -- driver ----------------------------------------------------------- #
    def run(self, start: Optional[str] = None):
        dispatch = {
            self.BUILD_DOMAINS: self._build_domains,
            self.ALL_GROUP: self._all_group,
            self.MPS: self._mps,
            self.PROFILER: self._profiler,
            self.SERVER_ENV: self._server_env,
            self.SUMMARY: self._summary,
            self.EDIT_MENU: self._edit_menu,
            self.EDIT_DOMAINS: self._edit_domains,
        }
        state = start or self.BUILD_DOMAINS
        while state not in (self.DONE, self.ABORT):
            state = dispatch[state]()
        return self.config if state == self.DONE else None

    def _advance(self, current):
        """Next state for a linear-or-edit phase."""
        return self.SUMMARY if self.editing else self._LINEAR_NEXT[current]

    # -- inventory helpers ------------------------------------------------ #
    def _used_gpu_aliases(self):
        out = set()
        for d in self.config.domains:
            g = identity_mod.resolve_gpu(self.inv, d.gpu_uuid)
            if g:
                out.add(g.alias)
        return out

    def _used_drive_aliases(self, exclude_domain=None):
        out = set()
        for d in self.config.domains:
            if exclude_domain is not None and d.name == exclude_domain:
                continue
            for dr in d.drives:
                rd = identity_mod.resolve_drive(self.inv, dr.serial or dr.pci_bus_id)
                if rd:
                    out.add(rd.alias)
        return out

    def _drive_display_name(self, dr):
        rd = identity_mod.resolve_drive(self.inv, dr.serial or dr.pci_bus_id)
        drive = rd or dr
        linux_device = getattr(drive, "linux_device", None)
        return os.path.basename(linux_device) if linux_device else (
            getattr(drive, "alias", None) or dr.serial or dr.pci_bus_id
        )

    def _find_domain_by_gpu(self, token):
        g = identity_mod.resolve_gpu(self.inv, token)
        if g is None:
            return None, None
        for d in self.config.domains:
            if d.gpu_uuid == g.uuid:
                return d, g
        return None, g

    # -- display helpers -------------------------------------------------- #
    def _print_gpu_table(self, gpus):
        print("Available GPUs:")
        for g in gpus:
            print(f"  {g.alias:6} smi={str(g.smi_index):<3} "
                  f"{(g.name or ''):24} {(g.pci_bus_id or ''):16} {g.policy_label()}")
        print()

    def _print_drive_table(self, drives, g):
        print("Unassigned NVMe (host device name; status is advisory; apply enforces):")
        for d in drives:
            name = self._drive_display_name(d)
            print(f"  {name:12} {(d.pci_bus_id or ''):16} "
                  f"{(d.model or ''):20} {(d.size or ''):8} {d.status_label()}")
        print(f"  for GPU: {g.alias}, smi={g.smi_index}, "
              f"{g.name or ''}, {g.pci_bus_id or ''}")
        print()

    def _print_domain_list(self):
        print("Current domains:")
        for d in self.config.domains:
            g = identity_mod.resolve_gpu(self.inv, d.gpu_uuid)
            alias = g.alias if g else d.gpu_uuid
            drv = ", ".join(self._drive_display_name(dr) for dr in d.drives)
            print(f"  {d.name}: {alias} + {drv}")
        print()

    # -- domain building -------------------------------------------------- #
    def _select_drives(self, g, exclude_domain=None):
        """Prompt for a GPU's drives. Returns a list of Drives, or None if the
        user enters a blank line (cancel)."""
        while True:
            used = self._used_drive_aliases(exclude_domain)
            avail = [d for d in self.inv.drives if d.alias not in used]
            if not avail:
                print("  No unassigned drives remain.\n")
                return None
            self._print_drive_table(avail, g)
            raw = _prompt(
                f"Select drives for {g.alias} (comma-separated device names, blank to cancel)", ""
            )
            if not raw:
                return None
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            resolved, seen, ok = [], set(), True
            for t in tokens:
                d = identity_mod.resolve_drive(self.inv, t)
                if d is None:
                    print(f"    '{t}' is not a known drive; try again.\n")
                    ok = False
                    break
                if d.alias in used or d.alias in seen:
                    print(f"    {self._drive_display_name(d)} is already assigned; try again.\n")
                    ok = False
                    break
                if d.available is not True:
                    msg = (f"    WARNING: {self._drive_display_name(d)} is not available "
                           f"({d.availability_label()}); selecting it anyway.")
                    if sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == "":
                        msg = f"\033[33m{msg}\033[0m"
                    print(f"{msg}\n")
                seen.add(d.alias)
                resolved.append(d)
            if not ok or not resolved:
                continue
            backends = {d.storage_backend or "kernel" for d in resolved}
            if len(backends) != 1:
                print("    selected drives use different storage backends; "
                      "choose drives for one backend only.\n")
                continue
            config_backends = {
                d.storage_backend for d in self.config.domains
                if d.name != exclude_domain
            }
            if config_backends and backends != config_backends:
                print("    this config already uses the "
                      f"{next(iter(config_backends))} backend; installed server "
                      "packages are mutually exclusive.\n")
                continue
            return resolved

    def _socket_group(self, default=None):
        while True:
            group = _prompt("Client socket group ('-' for root only)", default or "").strip()
            if group == "-":
                return None
            if not group or config_mod._group_exists(group):
                return group or None
            print(f"  group '{group}' does not exist; create it first or choose an existing group.")

    def _add_domain_for_gpu(self, g):
        drives = self._select_drives(g)
        if drives is None:
            print(f"  No NVMe selected — returning to GPU selection; "
                  f"no domain created for {g.alias}.\n")
            return
        name = f"{g.alias}_domain"
        try:
            config_mod.add_domain(
                self.config, self.inv, name=name, gpu_token=g.alias,
                drive_tokens=[d.alias for d in drives],
                socket_group=self._socket_group(),
            )
        except ValueError as exc:
            print(f"  error: {exc}\n")
            return
        print(f"  + {name} ({drives[0].storage_backend or 'kernel'}): {g.alias} + "
              f"{', '.join(self._drive_display_name(d) for d in drives)}\n")

    def _domain_build_loop(self):
        """Pick a GPU, then its drives; repeat. Blank GPU finishes (with a
        confirm, and a guard requiring at least one domain)."""
        while True:
            avail = [g for g in self.inv.gpus if g.alias not in self._used_gpu_aliases()]
            if not avail:
                print("All GPUs are assigned.\n")
                return
            self._print_gpu_table(avail)
            tok = _prompt("Select GPU (alias, blank to finish)", "")
            if not tok:
                if not self.config.domains:
                    print("No domains defined yet — add at least one "
                          "(or press Ctrl-C to cancel).\n")
                    continue
                if _prompt_yes_no("Done defining domains?", default=True):
                    return
                continue
            g = identity_mod.resolve_gpu(self.inv, tok)
            if g is None or g.alias in self._used_gpu_aliases():
                print(f"  '{tok}' is not an available GPU; try again.\n")
                continue
            self._add_domain_for_gpu(g)

    def _build_domains(self):
        print("\nDefine domains: pick a GPU, then assign its drives.\n")
        self._domain_build_loop()
        return self.ALL_GROUP

    # -- settings phases -------------------------------------------------- #
    def _all_group(self):
        if len(self.config.domains) >= 2:
            if _prompt_yes_no(
                "Create an 'all' group containing every domain?", default=True
            ):
                self.config.domain_groups["all"] = [
                    d.name for d in self.config.domains
                ]
            else:
                self.config.domain_groups.pop("all", None)
        else:
            self.config.domain_groups.pop("all", None)
            if self.editing:
                print("The 'all' group only applies with 2+ domains; skipping.")
        return self._advance(self.ALL_GROUP)

    def _mps(self):
        was_enabled = self.config.mps_enabled
        self.config.mps_enabled = _prompt_yes_no("Enable MPS?", default=self.config.mps_enabled)
        if self.config.mps_enabled:
            print("Root servers with non-root GPU clients require multiuser MPS.")
            print("Multiuser MPS removes user isolation within this MPS instance; use only with trusted users.")
            self.config.mps_multiuser = _prompt_yes_no(
                "Enable multiuser MPS?", default=self.config.mps_multiuser if was_enabled else True
            )
        else:
            self.config.mps_multiuser = False
        return self._advance(self.MPS)

    def _profiler(self):
        if len(self.config.domains) != 1:
            self.config.profiler_endpoint_mode = PROFILER_ENDPOINT_MODE_MANAGED
            print("Profiler endpoint mode: managed (one unique socket per GPU domain).")
            print("Nsight legacy compatibility is unavailable for multi-GPU configurations.")
            return self._advance(self.PROFILER)

        print("Profiler endpoint mode:")
        print("  managed     unique /var/run/scada endpoint (recommended)")
        print("  legacy_nsys /tmp/scada_profiler_socket for the current Nsight plugin")
        while True:
            mode = _prompt(
                "Choose managed or legacy_nsys",
                self.config.profiler_endpoint_mode,
            )
            if mode in (PROFILER_ENDPOINT_MODE_MANAGED, PROFILER_ENDPOINT_MODE_LEGACY_NSYS):
                self.config.profiler_endpoint_mode = mode
                return self._advance(self.PROFILER)
            print("  please choose managed or legacy_nsys")

    def _server_env(self):
        if not _prompt_yes_no(
            f"Keep these server env values ({_fmt_env(self.config.server_env)})?",
            default=True,
        ):
            self.config.server_env = _prompt_server_env(self.config.server_env)
        return self._advance(self.SERVER_ENV)

    # -- summary / confirm / edit ---------------------------------------- #
    def _sync_all_group(self):
        """Keep an existing 'all' group consistent with the current domains."""
        if "all" in self.config.domain_groups:
            if len(self.config.domains) >= 2:
                self.config.domain_groups["all"] = [
                    d.name for d in self.config.domains
                ]
            else:
                self.config.domain_groups.pop("all", None)
        if (
            self.config.profiler_endpoint_mode == PROFILER_ENDPOINT_MODE_LEGACY_NSYS
            and len(self.config.domains) != 1
        ):
            if _prompt_yes_no(
                "Nsight legacy compatibility permits one domain. Switch to managed endpoints?",
                default=True,
            ):
                self.config.profiler_endpoint_mode = PROFILER_ENDPOINT_MODE_MANAGED

    def _print_summary(self):
        print(f"\nConfig: {self.config.name}\n")
        for d in self.config.domains:
            g = identity_mod.resolve_gpu(self.inv, d.gpu_uuid)
            print(d.name)
            if g:
                print(f"  GPU: {g.alias}, {g.name or ''}, {g.pci_bus_id or ''}")
            else:
                print(f"  GPU: {d.gpu_uuid}, {d.gpu_pci_bus_id}")
            print(f"  Socket group: {d.socket_group or '(root only)'}")
            for dr in d.drives:
                rd = identity_mod.resolve_drive(self.inv, dr.serial or dr.pci_bus_id)
                if rd:
                    print(f"  NVME: {self._drive_display_name(rd)}, "
                          f"{rd.model or ''}, {rd.pci_bus_id or ''}")
                else:
                    print(f"  NVME: {dr.serial or ''}, , {dr.pci_bus_id or ''}")
        print()
        grp = self.config.domain_groups.get("all")
        print(f"all group: {', '.join(grp) if grp else '(none)'}")
        print(f"MPS: {'enabled' if self.config.mps_enabled else 'disabled'}")
        if self.config.mps_enabled:
            print(f"MPS multiuser: {self.config.mps_multiuser}")
        print(f"Profiler endpoint mode: {self.config.profiler_endpoint_mode}")
        print(f"server_env: {_fmt_env(self.config.server_env)}")

    def _summary(self):
        self.editing = True
        self._sync_all_group()
        self._print_summary()
        # Non-blocking validation warnings (resource-in-use etc. don't block).
        res = config_mod.validate_config(
            self.config, inventory=self.inv, registry=registry_mod.read_registry()
        )
        for w in res.warnings:
            print(f"  warning: {w}")
        if res.errors:
            for e in res.errors:
                print(f"  error: {e}")
            return self.EDIT_MENU
        if _prompt_yes_no("\nWrite this config?", default=True):
            return self.DONE
        return self.EDIT_MENU

    def _edit_menu(self):
        print("\nWhat would you like to edit?")
        print("  1) domains")
        print("  2) all group")
        print("  3) MPS")
        print("  4) server env")
        print("  5) profiler endpoint mode")
        choice = _prompt("Choose 1-5", "1")
        return {
            "1": self.EDIT_DOMAINS,
            "2": self.ALL_GROUP,
            "3": self.MPS,
            "4": self.SERVER_ENV,
            "5": self.PROFILER,
        }.get(choice, self.EDIT_MENU)

    def _edit_domains(self):
        print("\nEdit domains: [a]dd / [m]odify / [r]emove")
        action = _prompt("Choose", "add").strip().lower()
        if action in ("a", "add", ""):
            self._domain_build_loop()
        elif action in ("m", "modify"):
            self._modify_domain()
        elif action in ("r", "remove"):
            self._remove_domains()
        else:
            print(f"  unknown choice '{action}'.")
        return self.SUMMARY

    def _modify_domain(self):
        if not self.config.domains:
            print("  no domains to modify.")
            return
        self._print_domain_list()
        tok = _prompt("Modify which domain (GPU alias)", "")
        dom, g = self._find_domain_by_gpu(tok)
        if dom is None:
            print(f"  no domain found for GPU '{tok}'.")
            return
        drives = self._select_drives(g, exclude_domain=dom.name)
        if drives is None:
            print(f"  no change made to {dom.name}.")
            return
        self.config.domains = [d for d in self.config.domains if d.name != dom.name]
        config_mod.add_domain(
            self.config, self.inv, name=dom.name, gpu_token=g.alias,
            drive_tokens=[d.alias for d in drives],
            socket_group=self._socket_group(dom.socket_group),
            server_env=dom.server_env,
        )
        print(f"  updated {dom.name}: {g.alias} + "
              f"{', '.join(self._drive_display_name(d) for d in drives)}")

    def _remove_domains(self):
        while self.config.domains:
            self._print_domain_list()
            tok = _prompt("Remove which domain (GPU alias, blank to finish)", "")
            if not tok:
                return
            dom, _g = self._find_domain_by_gpu(tok)
            if dom is None:
                print(f"  no domain found for GPU '{tok}'.")
                continue
            self.config.domains = [
                d for d in self.config.domains if d.name != dom.name
            ]
            print(f"  removed {dom.name}")
        print("  no domains remain.")
