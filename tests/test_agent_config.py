from __future__ import annotations

from pathlib import Path

from agent.config import get_provider_config


def test_provider_config_uses_yaml_and_model_override(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        """
default_provider: qwen
providers:
  qwen:
    api_key: yaml-key
    base_url: https://example.test/v1
    model: qwen-test
    api_type: openai_compatible
""",
        encoding="utf-8",
    )

    provider = get_provider_config(provider="qwen", model_name="override-model", config_path=cfg)

    assert provider.name == "qwen"
    assert provider.api_key == "yaml-key"
    assert provider.base_url == "https://example.test/v1"
    assert provider.model == "override-model"


def test_provider_config_falls_back_to_env(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(
        """
default_provider: deepseek
providers:
  deepseek:
    api_key: ""
    base_url: ""
    model: ""
    api_type: openai_compatible
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example/v1")
    monkeypatch.setenv("CAD_AGENT_DEEPSEEK_MODEL", "deepseek-test")

    provider = get_provider_config(config_path=cfg)

    assert provider.name == "deepseek"
    assert provider.api_key == "env-key"
    assert provider.base_url == "https://deepseek.example/v1"
    assert provider.model == "deepseek-test"
