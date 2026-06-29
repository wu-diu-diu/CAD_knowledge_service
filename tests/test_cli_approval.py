from __future__ import annotations

from agent.approval import ApprovalGate, cli_approval_handler


def _gate() -> ApprovalGate:
    return ApprovalGate(stage="layout", room_name="测试房间", summary={"lamp_count": 2}, snapshot_id="snap_0001")


def test_cli_approval_continue(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert cli_approval_handler(_gate(), state=None) == "continue"


def test_cli_approval_retry(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "重新")
    assert cli_approval_handler(_gate(), state=None) == "retry"


def test_cli_approval_exit(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "exit")
    assert cli_approval_handler(_gate(), state=None) == "exit"
