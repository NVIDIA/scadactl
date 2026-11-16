# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime registry: atomic, locked read/write to ``registry.json``.

Contract:

* Single file at ``<runtime_dir>/registry.json``, root-owned ``0644``,
  world-readable, writable only by root.
* **Writers** serialize on an ``flock`` over ``<runtime_dir>/registry.lock``,
  read the current registry, reconcile only the configs/domains they manage,
  then write a complete replacement file via write-temp + fsync + ``os.rename``
  + directory fsync. Writers never edit in place.
* **Readers** are lock-free; at worst they see the old or new complete file,
  never partial JSON.

The runtime directory is configurable via ``SCADA_RUNTIME_DIR`` (default
``/var/run/scada``) so tests can point it at a temp dir without root.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from scadactl.models import (
    Registry,
    RegistryDomain,
    ConfigEntry,
)

DEFAULT_RUNTIME_DIR = "/var/run/scada"
LEGACY_NSYS_PROFILER_SOCKET = "/tmp/scada_profiler_socket"
REGISTRY_FILENAME = "registry.json"
LOCK_FILENAME = "registry.lock"
HOST_LOCK_FILENAME = "host.lock"


def runtime_dir() -> str:
    """Return the runtime directory, honoring ``SCADA_RUNTIME_DIR``."""
    return os.environ.get("SCADA_RUNTIME_DIR", DEFAULT_RUNTIME_DIR)


def registry_path() -> str:
    return os.path.join(runtime_dir(), REGISTRY_FILENAME)


def lock_path() -> str:
    return os.path.join(runtime_dir(), LOCK_FILENAME)


def host_lock_path() -> str:
    return os.path.join(runtime_dir(), HOST_LOCK_FILENAME)


def socket_for_domain(gpu_alias: str, rt_dir: Optional[str] = None) -> str:
    """Compose the socket path the server publishes for a GPU alias."""
    rt_dir = rt_dir or runtime_dir()
    return os.path.join(rt_dir, f"scadad.swap.{gpu_alias}.sock")


def profiler_socket_for_domain(gpu_alias: str, rt_dir: Optional[str] = None) -> str:
    """Compose scadactl's managed profiler endpoint for a GPU alias."""
    rt_dir = rt_dir or runtime_dir()
    return os.path.join(rt_dir, f"scadad.profiler.{gpu_alias}.sock")


# --------------------------------------------------------------------------- #
# Reads (lock-free)
# --------------------------------------------------------------------------- #
def _read_registry_file() -> Registry:
    """Parse the registry file (no lock). Empty registry if it is absent."""
    path = registry_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return Registry.empty()
    return Registry.from_dict(data)


def read_registry() -> Registry:
    """Read the registry without taking the writer lock.

    Returns an empty registry if the file is absent. Because writers publish
    atomically via rename, a concurrent read always sees a complete file.
    """
    return _read_registry_file()


# --------------------------------------------------------------------------- #
# Locked writes
# --------------------------------------------------------------------------- #
@contextmanager
def _flock(path: str) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` for the duration of the block.

    Lazy-imports ``fcntl`` so the module imports on non-POSIX platforms; the
    write path is POSIX-only in practice.
    """
    import fcntl

    os.makedirs(runtime_dir(), exist_ok=True)
    lf = open(path, "w", encoding="utf-8")
    try:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        finally:
            lf.close()


@contextmanager
def _writer_lock() -> Iterator[None]:
    """Serialize registry writers on an exclusive flock over ``registry.lock``."""
    with _flock(lock_path()):
        yield


@contextmanager
def host_lock() -> Iterator[None]:
    """Serialize whole privileged host operations (apply/stop) on ``host.lock``.

    A single coarse mutex -- distinct from the registry writer lock -- held for
    an operation's entire ``validate -> reclaim/bind/MPS/launch -> publish`` (or
    teardown) span, so two concurrent applies cannot both observe a resource as
    free and race on it (e.g. the loser's rollback unbinding a drive the winner
    now uses, or one apply reclaiming a drive the other just bound). It is a
    *separate* lock file from ``registry.lock`` on purpose: an apply holds this
    lock and still calls ``update_registry`` (which takes the writer lock) to
    publish, and re-locking one flock file from the same process would deadlock.
    Readers (``read_registry``, client selection, doctor) stay lock-free.
    """
    with _flock(host_lock_path()):
        yield


def _atomic_write(reg: Registry) -> None:
    """Write the registry atomically: temp + fsync + rename + dir fsync, 0644."""
    rt_dir = runtime_dir()
    os.makedirs(rt_dir, exist_ok=True)
    final = registry_path()
    tmp = final + ".tmp"
    payload = json.dumps(reg.to_dict(), indent=2, sort_keys=False)

    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    # 0644 explicitly (umask-independent).
    os.chmod(tmp, 0o644)
    os.rename(tmp, final)

    # fsync the directory so the rename is durable.
    dfd = os.open(rt_dir, os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def write_registry(reg: Registry) -> None:
    """Take the writer lock and publish a complete registry atomically."""
    with _writer_lock():
        _atomic_write(reg)


def update_registry(mutate: Callable[[Registry], Registry]) -> Registry:
    """Atomically read-modify-write the registry under the writer lock.

    Reads the *latest* on-disk registry inside the lock, applies ``mutate`` to
    produce the new registry, writes it atomically, and returns it. Because the
    read + reconcile + write is one critical section, two concurrent
    ``apply``/``stop`` invocations reconcile against each other's committed
    state rather than a stale snapshot, so neither can drop the other's domains.

    ``mutate`` runs while the lock is held; if it raises, the exception
    propagates (the lock is released) and nothing is written.
    """
    with _writer_lock():
        new = mutate(_read_registry_file())
        _atomic_write(new)
        return new


# --------------------------------------------------------------------------- #
# Reconcile semantics
# --------------------------------------------------------------------------- #
def _conflicting_domain_names(
    existing: Registry, config_name: str, new_domains: List[RegistryDomain]
) -> List[str]:
    """Names that already belong to a *different* config (hard conflict)."""
    owned_by_other = {
        d.name: d.config for d in existing.domains if d.config != config_name
    }
    return [d.name for d in new_domains if d.name in owned_by_other]


def reconcile_add_config(
    existing: Registry,
    config_name: str,
    new_domains: List[RegistryDomain],
    domain_groups: Optional[Dict[str, List[str]]] = None,
    mps_enabled: bool = False,
    mps_multiuser: bool = False,
) -> Registry:
    """Apply/reapply a config into the registry.

    * Domains owned by *other* configs are preserved untouched.
    * Any prior entries for ``config_name`` are replaced wholesale (reapply).
    * Raises ``ValueError`` if a new domain name collides with one owned by a
      different active config (domain names are globally unique).
    """
    conflicts = _conflicting_domain_names(existing, config_name, new_domains)
    if conflicts:
        raise ValueError(
            f"domain name(s) already owned by another config: {', '.join(sorted(set(conflicts)))}"
        )

    # Keep everything not owned by this config; drop this config's old entries.
    kept = [d for d in existing.domains if d.config != config_name]
    domains = kept + list(new_domains)

    configs = [p for p in existing.configs if p.name != config_name]
    configs.append(
        ConfigEntry(
            name=config_name,
            domains=[d.name for d in new_domains],
            domain_groups={k: list(v) for k, v in (domain_groups or {}).items()},
            mps_enabled=mps_enabled,
            mps_multiuser=mps_multiuser,
        )
    )

    active = [p for p in existing.active_configs if p != config_name]
    active.append(config_name)

    return Registry(
        active_configs=active,
        configs=configs,
        socket_dir=existing.socket_dir,
        domains=domains,
    )


def reconcile_remove_config(existing: Registry, config_name: str) -> Registry:
    """Remove only ``config_name``'s entries (``stop --config``)."""
    domains = [d for d in existing.domains if d.config != config_name]
    configs = [p for p in existing.configs if p.name != config_name]
    active = [p for p in existing.active_configs if p != config_name]
    return Registry(
        active_configs=active,
        configs=configs,
        socket_dir=existing.socket_dir,
        domains=domains,
    )


def reconcile_remove_all(existing: Registry) -> Registry:
    """Clear all managed entries (``stop --all``)."""
    return Registry(socket_dir=existing.socket_dir)


def reconcile_remove_domains(
    existing: Registry, removed: List[Tuple[str, str]]
) -> Registry:
    """Remove specific ``(config, domain)`` pairs, keeping everything else.

    Domains not listed are preserved untouched. Each config entry's domain list
    and groups are pruned to the domains that remain; a config left with no
    domains is dropped entirely (and removed from ``active_configs``). Used by
    the partial-teardown paths in ``stop`` so a domain that could not be stopped
    stays tracked.
    """
    removed_set = set(removed)
    domains = [d for d in existing.domains if (d.config, d.name) not in removed_set]

    remaining: Dict[str, set] = {}
    for d in domains:
        remaining.setdefault(d.config, set()).add(d.name)

    configs = []
    for c in existing.configs:
        names = remaining.get(c.name)
        if not names:
            continue
        configs.append(
            ConfigEntry(
                name=c.name,
                domains=[n for n in c.domains if n in names],
                domain_groups={
                    g: [m for m in members if m in names]
                    for g, members in c.domain_groups.items()
                },
                # Preserve the original MPS flag: a partial stop must not silently
                # clear it, or a later `stop --all` would skip stopping the daemon.
                mps_enabled=c.mps_enabled,
                mps_multiuser=c.mps_multiuser,
            )
        )
    active = [c for c in existing.active_configs if c in remaining]
    return Registry(
        active_configs=active,
        configs=configs,
        socket_dir=existing.socket_dir,
        domains=domains,
    )
