# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Init wizard: early config-name collision warning + rename loop."""

from __future__ import annotations

from scadactl import wizard as wizard_mod


def _seq_prompt(responses):
    it = iter(responses)

    def _fake(text, default=None):
        try:
            return next(it)
        except StopIteration:
            return default or ""

    return _fake


def _seq_yesno(answers):
    it = iter(answers)

    def _fake(text, default=False):
        try:
            return next(it)
        except StopIteration:
            return default

    return _fake


def test_name_no_collision_returns_immediately(monkeypatch):
    monkeypatch.setattr(wizard_mod, "_prompt", _seq_prompt(["fresh"]))
    monkeypatch.setattr(wizard_mod.store_mod, "exists", lambda n: False)
    assert wizard_mod._prompt_config_name("fresh", force=False) == "fresh"


def test_name_collision_then_rename_to_free(monkeypatch, capsys):
    # 'demo' exists; user opts to pick a different name -> 'demo2' (free).
    monkeypatch.setattr(wizard_mod.store_mod, "exists", lambda n: n == "demo")
    monkeypatch.setattr(wizard_mod, "_prompt", _seq_prompt(["demo", "demo2"]))
    monkeypatch.setattr(wizard_mod, "_prompt_yes_no", _seq_yesno([True]))
    assert wizard_mod._prompt_config_name("demo", force=False) == "demo2"
    assert "already exists" in capsys.readouterr().err


def test_name_collision_keep_is_warned(monkeypatch, capsys):
    monkeypatch.setattr(wizard_mod.store_mod, "exists", lambda n: n == "demo")
    monkeypatch.setattr(wizard_mod, "_prompt", _seq_prompt(["demo"]))
    monkeypatch.setattr(wizard_mod, "_prompt_yes_no", _seq_yesno([False]))  # keep
    assert wizard_mod._prompt_config_name("demo", force=False) == "demo"
    err = capsys.readouterr().err
    assert "already exists" in err and "keeping 'demo'" in err


def test_repeated_collision_until_free(monkeypatch):
    taken = {"demo", "demo2"}
    monkeypatch.setattr(wizard_mod.store_mod, "exists", lambda n: n in taken)
    monkeypatch.setattr(wizard_mod, "_prompt", _seq_prompt(["demo", "demo2", "demo3"]))
    monkeypatch.setattr(wizard_mod, "_prompt_yes_no", _seq_yesno([True, True]))
    assert wizard_mod._prompt_config_name("demo", force=False) == "demo3"


def test_force_skips_collision_check_entirely(monkeypatch):
    called = {"n": 0}

    def _exists(n):
        called["n"] += 1
        return True

    monkeypatch.setattr(wizard_mod.store_mod, "exists", _exists)
    monkeypatch.setattr(wizard_mod, "_prompt", _seq_prompt(["demo"]))
    assert wizard_mod._prompt_config_name("demo", force=True) == "demo"
    assert called["n"] == 0  # --force means overwrite is intended; never probes


def test_store_has_never_aborts_on_store_error(monkeypatch):
    def _boom(n):
        raise PermissionError("store unreadable")

    monkeypatch.setattr(wizard_mod.store_mod, "exists", _boom)
    assert wizard_mod._store_has("x") is False
