"""Model providers. The memory engine works fully offline; a provider only
upgrades two things: fact distillation during consolidation, and (optionally)
embeddings. Nothing in the recall path requires a model.

Keep the interface tiny so any provider can slot in.
"""

from __future__ import annotations

import json
import os
import urllib.request


class Provider:
    name = "base"

    def complete(self, prompt):
        """Return a completion string for prompt. Raise on failure."""
        raise NotImplementedError


class StubProvider(Provider):
    """Offline stand-in. Returns empty so callers fall back to rule-based
    extraction. Deterministic: same input, same output, always."""

    name = "stub"

    def complete(self, prompt):
        return ""


class OpenAIProvider(Provider):
    """Minimal OpenAI chat client over urllib. No SDK dependency."""

    name = "openai"

    def __init__(self, model=None):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("TINYMEM_MODEL", "gpt-4o-mini")

    def complete(self, prompt):
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 300,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()


def default_provider():
    """OpenAI when a key is present, stub otherwise. Never fails."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIProvider()
        except RuntimeError:
            pass
    return StubProvider()
