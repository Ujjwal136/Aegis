from __future__ import annotations

from typing import Any

import httpx

from config import settings


class LLMAgent:
    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower().strip()

    def ask(self, prompt: str) -> str:
        if self.provider == "openai" and settings.openai_api_key:
            return self._ask_openai(prompt)
        if self.provider == "anthropic" and settings.anthropic_api_key:
            return self._ask_anthropic(prompt)
        return self._ask_mock(prompt)

    def synthesize(self, sanitized_db_data: Any, original_prompt: str) -> str:
        prompt = (
            "You are a banking assistant. Summarize sanitized data safely for the user.\n"
            f"User prompt: {original_prompt}\n"
            f"Sanitized data: {sanitized_db_data}"
        )
        return self.ask(prompt)

    def handle_blocked(self) -> str:
        return "Your request was blocked by Aegis Firewall due to potential prompt-injection or data-exfiltration risk."

    def _ask_openai(self, prompt: str) -> str:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=20.0) as client:
            response = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def _ask_anthropic(self, prompt: str) -> str:
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=20.0) as client:
            response = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["content"][0]["text"]

    def _ask_mock(self, prompt: str) -> str:
        if "Sanitized data:" in prompt:
            return prompt.split("Sanitized data:", maxsplit=1)[1].strip()
        return "Aegis mock reply: request received."
