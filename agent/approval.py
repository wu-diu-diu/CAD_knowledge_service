from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ApprovalGate:
    stage: str
    room_name: str
    summary: Dict[str, Any]
    snapshot_id: str
    options: List[str] = field(default_factory=lambda: ["continue", "retry", "exit"])


def cli_approval_handler(gate: ApprovalGate, state: Any) -> str:
    print("")
    print(f"[approval] stage={gate.stage} room={gate.room_name}")
    print(f"[approval] summary={gate.summary}")
    print("输入 y/yes/确认 继续，n/no/重新 回滚并重试，exit 结束本轮。")
    while True:
        answer = input("approval> ").strip().lower()
        if answer in {"y", "yes", "ok", "确认", "继续", ""}:
            return "continue"
        if answer in {"n", "no", "retry", "重新", "重试"}:
            return "retry"
        if answer in {"exit", "quit", "q", "退出"}:
            return "exit"
        print("无法识别，请输入 y、n 或 exit。")
