"""Tests for the proposer abstraction layer."""

from __future__ import annotations

from src.proposers import MockProposer
from src.proposers.base import Proposer


def test_mock_proposer_returns_string():
    mp = MockProposer(response="hi")
    assert mp.generate("sys", "user") == "hi"
    assert mp.calls[0]["system"] == "sys"
    assert mp.calls[0]["user"] == "user"


def test_mock_proposer_callable():
    mp = MockProposer(response=lambda s, u: f"echo: {u[:5]}")
    assert mp.generate("s", "hello world") == "echo: hello"


def test_mock_proposer_iterable_responses():
    """A list of responses serves them in order, sticking on the last when exhausted."""
    mp = MockProposer(response=["one", "two", "three"])
    assert mp.generate("", "") == "one"
    assert mp.generate("", "") == "two"
    assert mp.generate("", "") == "three"
    assert mp.generate("", "") == "three"  # sticks
    assert len(mp.calls) == 4


def test_proposer_protocol_compatibility():
    """MockProposer should structurally satisfy the Proposer protocol."""
    mp = MockProposer(response="x")
    # No isinstance check — Protocol uses structural typing. Just verify
    # it has the expected attribute and callable.
    assert hasattr(mp, "generate")
    assert hasattr(mp, "name")
    # Type-check the call signature can accept the keyword args
    out = mp.generate("s", "u", max_tokens=100, temperature=0.5)
    assert out == "x"
    assert mp.calls[-1]["max_tokens"] == 100
    assert mp.calls[-1]["temperature"] == 0.5
