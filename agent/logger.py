from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


class AgentRunLogger:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    GRAY = "\033[90m"

    def __init__(
        self,
        log_dir: Optional[str] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        root_dir = Path(__file__).resolve().parents[1]
        final_log_dir = Path(log_dir) if log_dir else (root_dir / "logs")
        final_log_dir.mkdir(parents=True, exist_ok=True)
        self.max_line_len = max(120, int(os.getenv("CAD_AGENT_LOG_MAX_LEN", "320")))
        self.event_sink = event_sink
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = str(final_log_dir / f"react_agent_{ts}.log")
        self._emit("SESSION", f"log file: {self.log_path}", self.GRAY)
        self._publish(
            "session",
            {
                "log_file": self.log_path,
            },
        )

    @staticmethod
    def _safe_json(data: Any) -> str:
        try:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            return str(data)

    def _one_line(self, text: str, max_len: Optional[int] = None) -> str:
        limit = self.max_line_len if max_len is None else max(80, int(max_len))
        compact = re.sub(r"\s+", " ", str(text)).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _one_line_json(self, data: Any, max_len: Optional[int] = None) -> str:
        return self._one_line(self._safe_json(data), max_len=max_len)

    def _emit(self, tag: str, message: str, color: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        plain = f"[{ts}] [{tag}] {message}"
        colored = f"{color}{plain}{self.RESET}"
        print(colored + "\n\n", end="")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(plain + "\n\n")

    def _publish(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.event_sink is None:
            return
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
        }
        if isinstance(payload, dict):
            event.update(payload)
        try:
            self.event_sink(event)
        except Exception:
            # Streaming sink failures should not break the agent run.
            return

    def room_start(self, room_name: str, area_m2: float, shape: Tuple[int, int]) -> None:
        self._emit(
            "ROOM",
            f"start room='{room_name}' area_m2={float(area_m2):.3f} matrix_shape={list(shape)}",
            self.CYAN + self.BOLD,
        )
        self._publish(
            "room_start",
            {
                "room_name": room_name,
                "area_m2": float(area_m2),
                "matrix_shape": list(shape),
            },
        )

    def room_end(self, room_name: str, finish_reason: str, tool_calls: int, score: Optional[int]) -> None:
        self._emit(
            "ROOM",
            f"end room='{room_name}' finish='{finish_reason}' tool_calls={tool_calls} score={score}",
            self.CYAN + self.BOLD,
        )
        self._publish(
            "room_end",
            {
                "room_name": room_name,
                "finish_reason": finish_reason,
                "tool_calls": int(tool_calls),
                "score": score,
            },
        )

    def llm_response(self, content: str) -> None:
        self._emit("LLM", f"raw_response={self._one_line(content)}", self.YELLOW)
        self._publish(
            "llm_response",
            {
                "content": str(content),
            },
        )

    def action(self, action_obj: Dict[str, Any]) -> None:
        action = str(action_obj.get("action", ""))
        args = self._one_line_json(action_obj.get("args", {}), max_len=180)
        reason = self._one_line(str(action_obj.get("reason", "")), max_len=100)
        msg = f"action={action} args={args}"
        if reason:
            msg += f" reason='{reason}'"
        self._emit("ACTION", msg, self.MAGENTA)
        self._publish(
            "action",
            {
                "action": action,
                "args": action_obj.get("args", {}) if isinstance(action_obj, dict) else {},
                "reason": str(action_obj.get("reason", "")) if isinstance(action_obj, dict) else "",
                "raw": action_obj,
            },
        )

    def thought(self, text: str) -> None:
        self._emit("THOUGHT", self._one_line(text, max_len=180), self.YELLOW + self.BOLD)
        self._publish(
            "thought",
            {
                "content": str(text),
            },
        )

    def tool_io(self, tool_name: str, tool_input: Dict[str, Any], tool_result: str) -> None:
        in_line = self._one_line_json(tool_input, max_len=180)
        out_line = self._one_line(str(tool_result), max_len=220)
        self._emit("TOOL", f"{tool_name} input={in_line} result='{out_line}'", self.BLUE)
        self._publish(
            "tool",
            {
                "tool": str(tool_name),
                "input": tool_input,
                "result": str(tool_result),
            },
        )

    def error(self, message: str) -> None:
        self._emit("ERROR", message, self.RED)
        self._publish(
            "error",
            {
                "message": str(message),
            },
        )
