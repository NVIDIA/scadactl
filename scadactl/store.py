# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed config store: a CLI-owned directory of config YAMLs keyed by name.

Configs become name-addressable so users don't depend on keeping a copy of the
YAML file. The store lives at ``/etc/scada/configs`` by default (world-readable,
root-writable; ``/etc/scada`` holds persistent configuration).
Override with ``SCADA_CONFIG_DIR`` (used by tests).

Reads (``list``/``load``) are unprivileged. Writes (``save``/``delete``) need
write access to the store directory -- i.e. root on a default install -- and
raise :class:`StoreError` with an actionable message otherwise.
"""

from __future__ import annotations

import os
from typing import List

from scadactl import config as config_mod
from scadactl.models import Config

DEFAULT_CONFIG_DIR = "/etc/scada/configs"


class StoreError(RuntimeError):
    """Raised for store I/O problems (missing config, permission denied, …)."""


def config_dir() -> str:
    return os.environ.get("SCADA_CONFIG_DIR") or DEFAULT_CONFIG_DIR


def _validate_name(name: str) -> None:
    """Reject config names that would escape the store directory.

    A config name is a single filename; any path separator, a ``.``/``..``
    component, or a NUL is refused so ``scadactl config show '../policy'`` cannot
    read (nor ``config rm`` delete) files outside the store.
    """
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or os.path.basename(name) != name
    ):
        raise StoreError(
            f"invalid config name {name!r}: use a bare name without path "
            f"separators or '..'"
        )


def _path(name: str) -> str:
    _validate_name(name)
    return os.path.join(config_dir(), f"{name}.yaml")


def exists(name: str) -> bool:
    return os.path.isfile(_path(name))


def list_names() -> List[str]:
    """Return the sorted names of stored configs.

    An absent store directory is normal (no configs yet) -> ``[]``. A store that
    exists but cannot be listed is NOT normal: the store is world-readable by
    design, so a permission error means the directory was created with the wrong
    mode. Surface that instead of masking it as "no configs" (which contradicts a
    ``config init`` that still sees a config via a bare stat).
    """
    d = config_dir()
    try:
        return sorted(
            fn[: -len(".yaml")] for fn in os.listdir(d) if fn.endswith(".yaml")
        )
    except FileNotFoundError:
        return []
    except PermissionError as exc:
        raise StoreError(
            f"cannot read the config store {d}: {exc}. The store is meant to be "
            f"world-readable; an admin can repair it with: "
            f"sudo chmod -R a+rX {os.path.dirname(d) or d}"
        ) from exc
    except OSError:
        return []


def load(name: str) -> Config:
    path = _path(name)
    if not os.path.isfile(path):
        raise StoreError(
            f"no stored config named '{name}' (looked in {config_dir()}); "
            f"list with 'scadactl config list'"
        )
    try:
        return config_mod.load_config(path)
    except (OSError, ValueError) as exc:
        raise StoreError(f"cannot read stored config '{name}': {exc}") from exc


def save(config: Config, overwrite: bool = False) -> str:
    """Write ``config`` into the store keyed by its name. Returns the path.

    Raises :class:`StoreError` if it already exists and ``overwrite`` is False,
    or if the store directory is not writable (needs root on a default install).
    """
    if not config.name:
        raise StoreError("config has no name; cannot store it")
    path = _path(config.name)
    if os.path.isfile(path) and not overwrite:
        raise StoreError(
            f"config '{config.name}' already exists in the store; "
            f"pass --force to overwrite"
        )
    d = config_dir()
    try:
        os.makedirs(d, exist_ok=True)
        # The store is world-readable by design (unprivileged `config list`/`load`
        # read it). makedirs honors the umask, which on a host with a restrictive
        # umask (e.g. 077) would create a root-only directory that `config list`
        # cannot enumerate; force the intended mode. Best-effort so a chmod failure
        # on a dir we do not own never blocks an otherwise-successful write.
        try:
            os.chmod(d, 0o755)
        except OSError:
            pass
        config_mod.save_config(config, path)
    except PermissionError as exc:
        raise StoreError(
            f"cannot write to the config store {d} (need root?): {exc}"
        ) from exc
    except OSError as exc:
        raise StoreError(f"cannot write config to {path}: {exc}") from exc
    return path


def delete(name: str) -> None:
    path = _path(name)
    if not os.path.isfile(path):
        raise StoreError(f"no stored config named '{name}'")
    try:
        os.remove(path)
    except PermissionError as exc:
        raise StoreError(
            f"cannot delete '{name}' from the store (need root?): {exc}"
        ) from exc
    except OSError as exc:
        raise StoreError(f"cannot delete '{name}': {exc}") from exc
