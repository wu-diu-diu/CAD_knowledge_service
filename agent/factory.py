from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import LightingAgent
from .config import get_provider_config
from .model_clients import build_model_client
from .tools import LightingTools


def build_lighting_agent(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    config_path: Optional[str] = None,
    tools: Optional[LightingTools] = None,
    temperature: float = 0.0,
    init_mode: str = "llm",
    log_dir: Optional[str] = None,
    event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    **_: Any,
) -> LightingAgent:
    provider_config = get_provider_config(
        provider=provider,
        model_name=model_name,
        config_path=config_path,
    )
    model_client = build_model_client(provider_config)
    return LightingAgent(
        tools=tools,
        model_client=model_client,
        provider=provider_config.name,
        model_name=provider_config.model,
        temperature=temperature,
        init_mode=init_mode,
        log_dir=log_dir,
        event_sink=event_sink,
    )
