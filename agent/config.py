from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_AGENT_CONFIG_PATH = AGENT_DIR / "config.yaml"
EXAMPLE_AGENT_CONFIG_PATH = AGENT_DIR / "config.example.yaml"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    api_type: str


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _merge_provider_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    base_providers = dict((base or {}).get("providers") or {})
    override_providers = dict((override or {}).get("providers") or {})
    providers: Dict[str, Any] = {}
    for name in set(base_providers) | set(override_providers):
        providers[name] = {
            **dict(base_providers.get(name) or {}),
            **dict(override_providers.get(name) or {}),
        }
    merged.update(override or {})
    merged["providers"] = providers
    return merged


def load_agent_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    example = _read_yaml(EXAMPLE_AGENT_CONFIG_PATH)
    explicit_path = Path(config_path).expanduser() if config_path else None
    env_path = os.getenv("CAD_AGENT_CONFIG", "").strip()
    local_path = explicit_path or (Path(env_path).expanduser() if env_path else DEFAULT_AGENT_CONFIG_PATH)
    return _merge_provider_configs(example, _read_yaml(local_path))


def _env_for_provider(provider: str, field: str) -> str:
    prefix_map = {
        "qwen": "DASHSCOPE",
        "dashscope": "DASHSCOPE",
        "deepseek": "DEEPSEEK",
        "glm": "GLM",
        "minimax": "MINIMAX",
    }
    prefix = prefix_map.get(provider.lower(), provider.upper())
    if field == "model":
        return f"CAD_AGENT_{prefix}_MODEL"
    return f"{prefix}_{field.upper()}"


def get_provider_config(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    config_path: Optional[str | Path] = None,
) -> ProviderConfig:
    config = load_agent_config(config_path)
    resolved_provider = (
        provider
        or os.getenv("CAD_AGENT_PROVIDER", "").strip()
        or str(config.get("default_provider", "qwen"))
    ).strip().lower()
    providers = dict(config.get("providers") or {})
    if resolved_provider not in providers:
        raise ValueError(f"unsupported agent provider: {resolved_provider}")

    raw = dict(providers.get(resolved_provider) or {})
    api_key = (
        str(raw.get("api_key") or "").strip()
        or os.getenv(_env_for_provider(resolved_provider, "api_key"), "").strip()
    )
    base_url = (
        str(raw.get("base_url") or "").strip()
        or os.getenv(_env_for_provider(resolved_provider, "base_url"), "").strip()
    )
    model = (
        (model_name or "").strip()
        or os.getenv(_env_for_provider(resolved_provider, "model"), "").strip()
        or str(raw.get("model") or "").strip()
    )
    api_type = str(raw.get("api_type") or "openai_compatible").strip().lower()

    return ProviderConfig(
        name=resolved_provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        api_type=api_type,
    )
