"""Claude proposer wrapping claude-agent-sdk subscription OAuth.

Used as the default during the refactor so the legacy researcher.py path
keeps working unchanged. The new four-phase worker uses LocalMLXProposer
instead.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional


class ClaudeProposer:
    name: str

    def __init__(self, model: str = "claude-opus-4-7"):
        self.model = model
        self.name = model

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> str:
        # Lazy import to keep tests / mock paths cheap.
        from claude_agent_sdk import (  # type: ignore
            AssistantMessage, ClaudeAgentOptions, TextBlock, query,
        )

        async def _run() -> str:
            options = ClaudeAgentOptions(
                model=self.model,
                system_prompt=system,
                max_turns=1,
                permission_mode="bypassPermissions",
                allowed_tools=[],
            )
            chunks: list[str] = []
            async for msg in query(prompt=user, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)
            return "".join(chunks).strip()

        # Same thread-trampoline trick as ClaudeJudge: run in a worker thread
        # so a Jupyter event loop doesn't trip asyncio.run().
        result: dict = {}

        def worker() -> None:
            try:
                result["value"] = asyncio.run(_run())
            except BaseException as e:
                result["error"] = e

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join()
        if "error" in result:
            raise result["error"]
        return result["value"]
