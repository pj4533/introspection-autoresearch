"""Mock proposer for tests and integration smoke runs.

Pass a fixed text or a callable; the mock returns it. Tracks every call so
tests can assert on prompt construction.
"""

from __future__ import annotations

from typing import Callable, Optional, Union


class MockProposer:
    name: str = "mock"

    def __init__(
        self,
        response: Union[str, Callable[[str, str], str], list[str]] = "",
    ):
        self._response = response
        self._idx = 0
        self.calls: list[dict] = []

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append({
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if callable(self._response):
            return self._response(system, user)
        if isinstance(self._response, list):
            i = min(self._idx, len(self._response) - 1)
            self._idx += 1
            return self._response[i]
        return self._response
