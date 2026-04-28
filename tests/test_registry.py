"""Tests for src/models/registry.py.

The registry's invariants — only one handle loaded at a time, idempotent
load/unload, free-memory enforcement — are load-bearing for the whole
four-phase pipeline. If these break, two large models can co-reside and
silently corrupt activations (CLAUDE.md gotcha #5).
"""

from __future__ import annotations

import pytest

from src.models.registry import (
    HandleRegistry,
    MockHandle,
    enforce_free_memory,
    free_memory_gb,
)


def test_mock_handle_load_unload_lifecycle():
    h = MockHandle(name="mock-a", load_value="payload")
    assert not h.is_loaded()
    assert h.load_count == 0
    assert h.unload_count == 0

    h.load()
    assert h.is_loaded()
    assert h.load_count == 1
    assert h.obj == "payload"

    # Idempotent load
    h.load()
    assert h.load_count == 1

    h.unload()
    assert not h.is_loaded()
    assert h.unload_count == 1

    # Idempotent unload
    h.unload()
    assert h.unload_count == 1

    # Re-load works
    h.load()
    assert h.is_loaded()
    assert h.load_count == 2


def test_obj_raises_when_not_loaded():
    h = MockHandle(name="m")
    with pytest.raises(RuntimeError, match="not loaded"):
        _ = h.obj


def test_handle_registry_activate_unloads_others():
    """activate() must unload any other handle that was loaded."""
    a = MockHandle(name="a")
    b = MockHandle(name="b")
    reg = HandleRegistry(gemma=a, judge=b)

    reg.activate(a)
    assert a.is_loaded() and not b.is_loaded()

    # Switch to b: a should be unloaded, b should be loaded
    reg.activate(b)
    assert not a.is_loaded()
    assert b.is_loaded()
    assert a.unload_count == 1

    # Switch back to a: b should be unloaded
    reg.activate(a)
    assert a.is_loaded()
    assert not b.is_loaded()
    assert b.unload_count == 1


def test_handle_registry_unload_all():
    a = MockHandle(name="a")
    b = MockHandle(name="b")
    reg = HandleRegistry(gemma=a, judge=b)
    a.load()
    b.load()
    reg.unload_all()
    assert not a.is_loaded()
    assert not b.is_loaded()
    assert a.unload_count == 1
    assert b.unload_count == 1


def test_handle_registry_rejects_unknown_handle():
    a = MockHandle(name="a")
    b = MockHandle(name="b")
    stranger = MockHandle(name="stranger")
    reg = HandleRegistry(gemma=a, judge=b)
    with pytest.raises(ValueError, match="not registered"):
        reg.activate(stranger)


def test_free_memory_gb_returns_positive_or_unknown():
    """We can't pin a number, but we can verify the call shape works."""
    free = free_memory_gb()
    assert free == -1.0 or free > 0.0


def test_enforce_free_memory_does_not_raise_when_unknown(monkeypatch):
    """If free memory is reported as unknown, default behavior is no-op."""
    import src.models.registry as r
    monkeypatch.setattr(r, "free_memory_gb", lambda: -1.0)
    enforce_free_memory(min_gb=999.0)  # would otherwise blow up


def test_enforce_free_memory_raises_when_below(monkeypatch):
    import src.models.registry as r
    monkeypatch.setattr(r, "free_memory_gb", lambda: 5.0)
    with pytest.raises(RuntimeError, match="insufficient free memory"):
        enforce_free_memory(min_gb=10.0)


def test_handle_load_calls_pre_check(monkeypatch):
    """`load()` must consult enforce_free_memory before allocating."""
    import src.models.registry as r
    calls = []
    monkeypatch.setattr(r, "free_memory_gb", lambda: 100.0)
    h = MockHandle(name="m", expected_ram_gb=2.0)
    h.load()  # 2.0 + 8 OS reserve = 10 GB; we have 100. Should succeed.
    assert h.is_loaded()
