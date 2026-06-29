from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentMemory:
    preferences: Dict[str, Any] = field(default_factory=dict)
    recent_designs: List[Dict[str, Any]] = field(default_factory=list)

    def remember_preference(self, key: str, value: Any) -> None:
        if value is not None:
            self.preferences[str(key)] = value

    def remember_design(self, room_name: str, summary: Dict[str, Any]) -> None:
        self.recent_designs.append({"room_name": room_name, "summary": dict(summary or {})})
        self.recent_designs = self.recent_designs[-20:]

    def get_preference(self, key: str, default: Optional[Any] = None) -> Any:
        return self.preferences.get(key, default)
