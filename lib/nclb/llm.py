"""LLM backend — supports Anthropic API and OpenAI-compatible local servers.

Abstracts the LLM call so the rest of the system doesn't care whether
it's talking to Claude, LM Studio, ollama, vLLM, or anything else.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from .voices import parse_json_response


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""

    # Backend: "anthropic" or "openai" (OpenAI-compatible, for local servers)
    backend: str = "openai"

    # For OpenAI-compatible (LM Studio, ollama, vLLM, etc.)
    base_url: str = "http://10.151.30.147:1234/v1"
    model: str = "local-model"           # LM Studio picks the loaded model
    api_key: str = "lm-studio"           # LM Studio doesn't check this

    # For Anthropic
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Generation params
    max_tokens: int = 4096
    temperature: float = 0.3             # lower = more deterministic


def create_llm(config: Optional[LLMConfig] = None) -> "LLMBackend":
    """Factory: create the right backend based on config."""
    if config is None:
        config = LLMConfig()

    if config.backend == "anthropic":
        return AnthropicBackend(config)
    else:
        return OpenAIBackend(config)


class LLMBackend:
    """Base class for LLM backends."""

    def complete(self, system: str, user_prompt: str) -> dict:
        """Send a single prompt, get back parsed JSON."""
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible backend (LM Studio, ollama, vLLM, etc.)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
            )
        except ImportError:
            # Fall back to raw HTTP if openai package not installed
            self.client = None

    def complete(self, system: str, user_prompt: str) -> dict:
        if self.client:
            return self._complete_sdk(system, user_prompt)
        return self._complete_http(system, user_prompt)

    def _complete_sdk(self, system: str, user_prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        text = response.choices[0].message.content
        try:
            return parse_json_response(text)
        except (json.JSONDecodeError, ValueError):
            return {"raw_response": text}

    def _complete_http(self, system: str, user_prompt: str) -> dict:
        """Raw HTTP fallback if openai package isn't installed."""
        import urllib.request

        payload = json.dumps({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }).encode()

        req = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())

        text = result["choices"][0]["message"]["content"]
        try:
            return parse_json_response(text)
        except (json.JSONDecodeError, ValueError):
            return {"raw_response": text}


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, config: LLMConfig):
        self.config = config
        import anthropic
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user_prompt: str) -> dict:
        response = self.client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        try:
            return parse_json_response(text)
        except (json.JSONDecodeError, ValueError):
            return {"raw_response": text}
