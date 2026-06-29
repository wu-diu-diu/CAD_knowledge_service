from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import anthropic
from openai import OpenAI

from .config import ProviderConfig
from .utils import extract_json


class ModelClient(Protocol):
    provider: str
    model: str
    base_url: str

    @property
    def has_credentials(self) -> bool:
        ...

    def complete_action(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        ...


class BaseModelClient:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._setup()

    def _setup(self) -> None:
        return

    @property
    def provider(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def has_credentials(self) -> bool:
        return bool(self.config.api_key)

    @staticmethod
    def _parsed_action(content: str) -> Dict[str, Any]:
        obj = extract_json(content)
        if isinstance(obj, dict) and obj.get("action"):
            obj["_raw_response"] = content
            return obj
        return {
            "action": "__parse_error__",
            "reason": "model_response_is_not_action_json",
            "raw_response": content,
        }


class OpenAICompatibleModelClient(BaseModelClient):
    def _setup(self) -> None:
        self._client: Optional[OpenAI] = None
        if self.has_credentials:
            self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url or None)

    def complete_action(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        if self._client is None:
            return {"action": "finish", "reason": "no_llm_key", "strategy": "deterministic tools only"}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = ((resp.choices or [{}])[0].message.content or "").strip()
        return self._parsed_action(content)


class AnthropicCompatibleModelClient(BaseModelClient):
    def _setup(self) -> None:
        self._client: Optional[anthropic.Anthropic] = None
        if self.has_credentials:
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url or None,
            )

    def complete_action(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        if self._client is None:
            return {"action": "finish", "reason": "no_llm_key", "strategy": "deterministic tools only"}
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        parts: List[str] = []
        for block in getattr(resp, "content", []) or []:
            text = str(getattr(block, "text", "") or "").strip()
            if text:
                parts.append(text)
        return self._parsed_action("\n".join(parts).strip())


class QwenModelClient(OpenAICompatibleModelClient):
    pass


class DeepSeekModelClient(OpenAICompatibleModelClient):
    pass


class GlmModelClient(OpenAICompatibleModelClient):
    pass


class MiniMaxModelClient(AnthropicCompatibleModelClient):
    pass


def build_model_client(config: ProviderConfig) -> ModelClient:
    if config.name in {"qwen", "dashscope"}:
        return QwenModelClient(config)
    if config.name == "deepseek":
        return DeepSeekModelClient(config)
    if config.name == "glm":
        return GlmModelClient(config)
    if config.name == "minimax":
        return MiniMaxModelClient(config)
    if config.api_type == "anthropic_compatible":
        return AnthropicCompatibleModelClient(config)
    return OpenAICompatibleModelClient(config)
