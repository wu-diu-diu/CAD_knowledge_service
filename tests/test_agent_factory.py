from __future__ import annotations

from pathlib import Path

from agent.base import LightingAgent
from agent.factory import build_lighting_agent
from agent.model_clients import DeepSeekModelClient, GlmModelClient, MiniMaxModelClient, QwenModelClient


def _config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        """
default_provider: qwen
providers:
  qwen:
    api_key: key
    base_url: https://qwen.example/v1
    model: qwen-test
    api_type: openai_compatible
  deepseek:
    api_key: key
    base_url: https://deepseek.example/v1
    model: deepseek-test
    api_type: openai_compatible
  glm:
    api_key: key
    base_url: https://glm.example/v1
    model: glm-test
    api_type: openai_compatible
  minimax:
    api_key: key
    base_url: https://minimax.example
    model: minimax-test
    api_type: anthropic_compatible
""",
        encoding="utf-8",
    )
    return cfg


def test_factory_builds_unified_agent_for_supported_providers(tmp_path: Path) -> None:
    cfg = _config_file(tmp_path)

    cases = {
        "qwen": QwenModelClient,
        "deepseek": DeepSeekModelClient,
        "glm": GlmModelClient,
        "minimax": MiniMaxModelClient,
    }
    for provider, expected_client in cases.items():
        agent = build_lighting_agent(provider=provider, config_path=str(cfg))
        assert isinstance(agent, LightingAgent)
        assert isinstance(agent.model_client, expected_client)
        assert agent.provider == provider
