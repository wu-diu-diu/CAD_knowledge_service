from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .approval import ApprovalGate
from .config import get_provider_config
from .logger import AgentRunLogger
from .model_clients import ModelClient, build_model_client
from .plan import AgentPlan
from .state import RoomAgentState
from .tools import LightingTools, REACT_SYSTEM_PROMPT_TEMPLATE, REACT_TOOLS
from .utils import describe_tool_result


ApprovalHandler = Callable[[ApprovalGate, RoomAgentState], str]


class LightingAgent:
    """
    统一的电气照明 ReAct Agent。

    所有模型厂商只替换 model_client；工具、状态、审批、计划和反思流程保持一致。
    """

    agent_name = "lighting-react-agent"

    def __init__(
        self,
        tools: Optional[LightingTools] = None,
        model_client: Optional[ModelClient] = None,
        provider: str = "qwen",
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        temperature: float = 0.0,
        init_mode: str = "llm",
        log_dir: Optional[str] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.tools = tools or LightingTools()
        self.temperature = float(temperature)
        self.init_mode = (init_mode or "llm").strip().lower()
        if self.init_mode not in ("rule", "llm"):
            self.init_mode = "llm"
        self.model_client = model_client or build_model_client(
            get_provider_config(provider=provider, model_name=model_name, config_path=config_path)
        )
        self.provider = self.model_client.provider
        self.model = self.model_client.model
        self.base_url = self.model_client.base_url
        self.run_logger = AgentRunLogger(log_dir=log_dir, event_sink=event_sink)
        self.plan = AgentPlan()
        self.run_logger._emit(
            "MODEL",
            f"provider={self.provider} model={self.model} base_url={self.base_url} init_mode={self.init_mode}",
            AgentRunLogger.GRAY,
        )

    @staticmethod
    def list_tools() -> List[Dict[str, Any]]:
        return deepcopy(REACT_TOOLS)

    def run_for_room(
        self,
        state: RoomAgentState,
        max_steps: int = 8,
        user_goal: Optional[str] = None,
        reset_layout: bool = True,
        plan_overrides: Optional[Dict[str, Any]] = None,
        skip_initial_design: bool = False,
        generate_wiring: bool = True,
        approval_handler: Optional[ApprovalHandler] = None,
    ) -> Dict[str, Any]:
        state.logger = self.run_logger
        self.run_logger.room_start(state.room_name, state.area_m2, state.matrix.shape)

        if reset_layout:
            state.placements["lamps"] = []
            state.placements["switches"] = []
            state.selected_lamp_type = None
            state.lamp_plan = None
            state.tool_cache = {}
            state.tool_history = []
            state.tool_result_history = []
            state.thought_history = []
            state.snapshots = {}
            state.snapshot_order = []

        init_stage = {
            "req": None,
            "count_plan": None,
            "flux_plan": None,
            "model_plan": None,
            "coords_plan": None,
            "validation": None,
        }
        final_reason = "max_steps_reached"
        latest_wiring: Optional[Dict[str, Any]] = None
        last_tool_output: Optional[Dict[str, Any]] = state.tool_cache.get("last_tool_result")

        if self.init_mode == "rule" and not skip_initial_design:
            before_layout_snapshot = state.snapshot("before_layout", "before deterministic initial layout")
            init_stage = self._run_replan_design(state, args=plan_overrides or {})
            last_tool_output = state.tool_cache.get("last_tool_result")
            approval_decision = self._request_approval(
                approval_handler=approval_handler,
                state=state,
                stage="layout",
                snapshot_id=before_layout_snapshot,
            )
            if approval_decision == "exit":
                final_reason = "approval_exit"
                return self._finalize_result(state, latest_wiring, final_reason, init_stage, generate_wiring=False)
            if approval_decision == "retry":
                state.restore(before_layout_snapshot)
                init_stage = self._run_replan_design(state, args=plan_overrides or {})
                last_tool_output = state.tool_cache.get("last_tool_result")
        else:
            first_validation = self.tools.tool_validate_layout(state)
            init_stage["validation"] = first_validation
            state.record(
                "init_mode",
                {
                    "mode": self.init_mode,
                    "skip_initial_design": bool(skip_initial_design),
                    "plan_overrides": plan_overrides or {},
                },
                {"status": "reuse_existing_layout" if skip_initial_design else "start_from_empty"},
            )
            last_tool_output = state.tool_cache.get("last_tool_result")

        available_tools = self.list_tools()
        valid_actions = {"finish"} | {
            str(item.get("function", {}).get("name", ""))
            for item in available_tools
            if isinstance(item, dict)
        }
        unknown_or_parse_errors = 0

        for _ in range(max(1, int(max_steps))):
            view = self.tools.tool_read_matrix_state(state)
            validation = self.tools.tool_validate_layout(state)
            action = self._decide_action(
                state=state,
                view=view,
                validation=validation,
                last_tool_output=last_tool_output,
                tools=available_tools,
                user_goal=user_goal,
            )
            if isinstance(action, dict) and action.get("thought"):
                self.run_logger.thought(str(action.get("thought")))
            self.run_logger.action(action if isinstance(action, dict) else {"action": "invalid", "raw": str(action)})
            self._record_thought(state, action)

            action_name = str((action or {}).get("action", "")).strip()
            args = (action or {}).get("args", {}) or {}
            if action_name == "finish":
                final_reason = str(action.get("reason", "done"))
                state.record("finish", {}, {"reason": final_reason, "strategy": action.get("strategy", "")})
                last_tool_output = state.tool_cache.get("last_tool_result")
                break

            if action_name not in valid_actions:
                unknown_or_parse_errors += 1
                self._record_tool_error(
                    state,
                    tool_name="__unknown_tool__",
                    tool_input={"action": action_name, "args": args},
                    message=f"未知工具: {action_name}",
                )
                last_tool_output = state.tool_cache.get("last_tool_result")
                if unknown_or_parse_errors >= 2:
                    final_reason = f"unknown_action:{action_name}"
                    break
                continue

            snapshot_id = ""
            if action_name == "tool_place_components":
                snapshot_id = state.snapshot("before_layout", "before tool_place_components")
            elif action_name == "tool_generate_wiring":
                snapshot_id = state.snapshot("before_wiring", "before tool_generate_wiring")

            if action_name == "tool_generate_wiring" and not generate_wiring:
                tool_output = {
                    "status": "skipped",
                    "reason": "generate_wiring_disabled",
                    "route_count": 0,
                    "routes": [],
                }
                state.record(
                    "skip_tool_generate_wiring",
                    {"requested_by_agent": True},
                    tool_output,
                    tool_result="当前运行禁用了 agent 侧布线，已跳过该工具调用。",
                )
            else:
                try:
                    tool_output = self.tools.invoke_tool(action_name, state, args)
                except Exception as exc:
                    unknown_or_parse_errors += 1
                    self._record_tool_error(state, action_name, args, f"工具执行失败: {exc}")
                    last_tool_output = state.tool_cache.get("last_tool_result")
                    if unknown_or_parse_errors >= 2:
                        final_reason = f"tool_failed:{action_name}"
                        break
                    continue
                self._sync_lamp_plan(state)
                if action_name == "tool_generate_wiring":
                    latest_wiring = tool_output

            unknown_or_parse_errors = 0
            self._record_reflection(state, action_name, tool_output)
            last_tool_output = state.tool_cache.get("last_tool_result")

            if action_name == "tool_place_components":
                after_snapshot = state.snapshot("after_layout", "after tool_place_components")
                decision = self._request_approval(approval_handler, state, "layout", snapshot_id or after_snapshot)
                if decision == "exit":
                    final_reason = "approval_exit"
                    break
                if decision == "retry" and snapshot_id:
                    state.restore(snapshot_id)
                    last_tool_output = state.tool_cache.get("last_tool_result")
            elif action_name == "tool_generate_wiring":
                after_snapshot = state.snapshot("after_wiring", "after tool_generate_wiring")
                decision = self._request_approval(approval_handler, state, "wiring", snapshot_id or after_snapshot)
                if decision == "exit":
                    final_reason = "approval_exit"
                    break
                if decision == "retry" and snapshot_id:
                    state.restore(snapshot_id)
                    latest_wiring = None
                    last_tool_output = state.tool_cache.get("last_tool_result")

        return self._finalize_result(state, latest_wiring, final_reason, init_stage, generate_wiring=generate_wiring)

    def _finalize_result(
        self,
        state: RoomAgentState,
        latest_wiring: Optional[Dict[str, Any]],
        final_reason: str,
        init_stage: Dict[str, Any],
        generate_wiring: bool,
    ) -> Dict[str, Any]:
        final_validation = self.tools.tool_validate_layout(state)
        if latest_wiring is None and generate_wiring and final_reason != "approval_exit":
            latest_wiring = self.tools.tool_generate_wiring(state=state)
        result = {
            "room_name": state.room_name,
            "selected_lamp_type": state.selected_lamp_type,
            "lamp_plan": state.lamp_plan,
            "placements": state.placements,
            "wiring_plan": latest_wiring,
            "log_file": self.run_logger.log_path,
            "tool_calls": len(state.tool_history),
            "finish_reason": final_reason,
            "strategy_summary": self._build_strategy_summary(state),
            "validation": final_validation,
            "stage_outputs": {
                "tool_lookup_room_requirement": init_stage.get("req") or state.tool_cache.get("room_requirement"),
                "tool_estimate_component_count": init_stage.get("count_plan") or state.tool_cache.get("component_count_plan"),
                "tool_calc_required_flux_per_lamp": init_stage.get("flux_plan") or state.tool_cache.get("flux_plan"),
                "tool_retrieve_lamp_model": init_stage.get("model_plan") or state.tool_cache.get("lamp_model_plan"),
                "tool_place_components": init_stage.get("coords_plan") or state.tool_cache.get("component_layout_plan"),
                "tool_generate_wiring": state.tool_cache.get("wiring_plan"),
                "initial_validation": init_stage.get("validation"),
            },
        }
        self.run_logger.room_end(
            room_name=state.room_name,
            finish_reason=final_reason,
            tool_calls=len(state.tool_history),
            score=(final_validation or {}).get("score"),
        )
        return result

    def _run_replan_design(self, state: RoomAgentState, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        req = self.tools.tool_lookup_room_requirement(state)
        count_plan = self.tools.tool_estimate_component_count(
            state=state,
            is_regular=args.get("is_regular"),
            min_spacing_m=float(args.get("min_spacing_m", 2.0)),
            max_spacing_m=float(args.get("max_spacing_m", 3.0)),
            max_lamps=int(args.get("max_lamps", 64)),
            switch_count=args.get("switch_count"),
        )
        lamp_count = max(1, int(args["lamp_count"])) if args.get("lamp_count") is not None else int(count_plan.get("lamp_count", 1))
        count_plan["lamp_count"] = lamp_count
        component_switch_count = (
            max(0, int(args["switch_count"]))
            if args.get("switch_count") is not None
            else int(count_plan.get("switch_count", 1))
        )
        count_plan["switch_count"] = component_switch_count
        target_lux = int(args.get("target_lux", req.get("target_lux", 300)))
        uf = float(args.get("uf", 0.6))
        mf = float(args.get("mf", 0.8))
        flux_plan = self.tools.tool_calc_required_flux_per_lamp(
            state=state,
            target_lux=target_lux,
            lamp_count=lamp_count,
            uf=uf,
            mf=mf,
        )
        lamp_type = str(args.get("lamp_type", req.get("lamp_type", "筒灯")))
        model_plan = self.tools.tool_retrieve_lamp_model(
            state=state,
            lamp_type=lamp_type,
            required_flux_lm=float(flux_plan.get("required_flux_per_lamp_lm", 1000.0)),
            top_k=int(args.get("top_k", 3)),
        )
        grid_rows = int(args.get("grid_rows", count_plan.get("grid_rows", 1)))
        grid_cols = int(args.get("grid_cols", count_plan.get("grid_cols", 1)))
        if args.get("grid_rows") is not None and args.get("grid_cols") is not None:
            count_plan["grid_rows"] = max(1, grid_rows)
            count_plan["grid_cols"] = max(1, grid_cols)
            count_plan["lamp_count"] = max(1, grid_rows * grid_cols)
            lamp_count = int(count_plan["lamp_count"])
        state.lamp_plan = {
            "room_name": state.room_name,
            "target_lux": int(target_lux),
            "grid_rows": int(count_plan.get("grid_rows", 1)),
            "grid_cols": int(count_plan.get("grid_cols", 1)),
            "lamp_count": int(lamp_count),
            "switch_count": int(component_switch_count),
            "required_flux_per_lamp_lm": float(flux_plan.get("required_flux_per_lamp_lm", 1000.0)),
            "selected_lamp": model_plan.get("selected_lamp", {}),
            "backup_options": model_plan.get("candidates", []),
            "spacing_m": float(count_plan.get("preferred_spacing_m", 2.4)),
            "uf": float(uf),
            "mf": float(mf),
            "is_regular": bool(count_plan.get("is_regular", True)),
        }
        state.tool_cache["lamp_plan"] = state.lamp_plan
        coords_plan = self.tools.tool_place_components(
            state=state,
            lamp_count=int(state.lamp_plan.get("lamp_count", 1)),
            switch_count=int(state.lamp_plan.get("switch_count", 1)),
            is_regular=bool(state.lamp_plan.get("is_regular", True)),
        )
        validation = self.tools.tool_validate_layout(state)
        state.record("internal_replan_design", {"args": args}, {"validation_score": validation.get("score")})
        self._record_reflection(state, "internal_replan_design", {"validation_score": validation.get("score")})
        return {
            "req": req,
            "count_plan": count_plan,
            "flux_plan": flux_plan,
            "model_plan": model_plan,
            "coords_plan": coords_plan,
            "validation": validation,
        }

    def _decide_action(
        self,
        state: RoomAgentState,
        view: Dict[str, Any],
        validation: Dict[str, Any],
        last_tool_output: Optional[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        user_goal: Optional[str],
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt(tools)
        user_prompt = self._build_user_prompt(state, view, validation, last_tool_output, user_goal=user_goal)
        try:
            action = self.model_client.complete_action(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                temperature=self.temperature,
            )
            raw = str(action.get("_raw_response", "") or action.get("raw_response", "") or "").strip()
            if raw:
                state.logger.llm_response(raw) if state.logger else None
            if action.get("action") != "__parse_error__":
                return action

            retry_prompt = "\n".join(
                [
                    user_prompt,
                    "",
                    "=== 上次输出无法解析 ===",
                    "请只输出一个合法 JSON 对象，必须包含 action 和 args。",
                    f"上次原文: {action.get('raw_response', '')}",
                ]
            )
            retry_action = self.model_client.complete_action(
                system_prompt=system_prompt,
                user_prompt=retry_prompt,
                tools=tools,
                temperature=self.temperature,
            )
            raw_retry = str(retry_action.get("_raw_response", "") or retry_action.get("raw_response", "") or "").strip()
            if raw_retry:
                state.logger.llm_response(raw_retry) if state.logger else None
            if retry_action.get("action") != "__parse_error__":
                return retry_action
        except Exception as exc:
            if state.logger:
                state.logger.error(f"llm request failed in _decide_action: {exc}")
        return {"action": "finish", "reason": "llm_parse_failed", "strategy": "fallback finish"}

    def _build_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        tool_names = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            fn = item.get("function", {}) or {}
            name = str(fn.get("name", "")).strip()
            if name:
                tool_names.append(name)
        return REACT_SYSTEM_PROMPT_TEMPLATE.format(
            tools_block=json.dumps(tools, ensure_ascii=False, indent=2),
            tool_names=", ".join(tool_names),
        )

    def _build_user_prompt(
        self,
        state: RoomAgentState,
        view: Dict[str, Any],
        validation: Dict[str, Any],
        last_tool_output: Optional[Dict[str, Any]] = None,
        user_goal: Optional[str] = None,
    ) -> str:
        history_window = 12
        summary = view.get("summary", {}) or {}
        placements = summary.get("placements", {}) or {}
        violations = validation.get("violations", []) or []
        suggestions = validation.get("suggestions", []) or []
        prompt_parts: List[str] = []

        if user_goal:
            prompt_parts.extend(["=== 用户任务 ===", str(user_goal).strip(), ""])

        prompt_parts.extend(
            [
                "=== 房间状态 ===",
                f"房间名: {summary.get('room_name', state.room_name)}",
                f"房间面积: {float(summary.get('area_m2', state.area_m2)):.2f}m2",
                f"矩阵尺寸: {summary.get('matrix_shape', [int(state.matrix.shape[0]), int(state.matrix.shape[1])])}",
                f"已选灯具: {summary.get('selected_lamp_type') or state.selected_lamp_type or '未选择'}",
                f"当前已放置灯具: {len(placements.get('lamps', []))}个",
                f"当前已放置开关: {len(placements.get('switches', []))}个",
                "",
                "=== ASCII 棋盘 ===",
                "#=障碍, .=可布置, D=门, L=灯, S=开关",
                view["ascii_board"],
                "",
                "=== 校验结果 ===",
                describe_tool_result("tool_validate_layout", validation),
            ]
        )
        if violations:
            prompt_parts.append("违规详情:")
            for item in violations[:4]:
                if isinstance(item, dict):
                    prompt_parts.append(f"- {item.get('code', 'UNKNOWN')}: {item.get('message', '')}")
        if suggestions:
            prompt_parts.append("修正建议:")
            for item in suggestions[:3]:
                prompt_parts.append(f"- {item}")
        if last_tool_output:
            prompt_parts.extend(
                [
                    "",
                    "=== 上轮工具执行 (Observation) ===",
                    f"工具: {last_tool_output.get('tool')}",
                    f"tool_result: {last_tool_output.get('tool_result', '')}",
                ]
            )
        if state.tool_result_history:
            prompt_parts.extend(["", "=== 近期工具结果 ==="])
            for idx, item in enumerate(state.tool_result_history[-history_window:], 1):
                prompt_parts.append(f"{idx}. {item.get('tool')}: {item.get('tool_result', '')}")
        if state.thought_history:
            prompt_parts.extend(["", "=== 近期决策历史 ==="])
            for idx, item in enumerate(state.thought_history[-8:], 1):
                prompt_parts.append(f"{idx}. {str(item.get('thought', ''))[:80]} [action={item.get('action')}]")
        reflections = state.tool_cache.get("reflections", []) if isinstance(state.tool_cache, dict) else []
        if reflections:
            prompt_parts.extend(["", "=== 最近反思 ==="])
            for item in reflections[-4:]:
                prompt_parts.append(f"- {item.get('phase')}: {item.get('next_hint')}")
        prompt_parts.extend(
            [
                "",
                "=== 请你的决策 ===",
                "输出 JSON 格式:",
                '{"thought": "简短理由", "action": "工具名", "args": {...}}',
            ]
        )
        return "\n".join(prompt_parts)

    @staticmethod
    def _build_strategy_summary(state: RoomAgentState) -> str:
        lamp_type = state.selected_lamp_type or "未知灯具"
        lamp_count = len(state.placements.get("lamps", []))
        switch_count = len(state.placements.get("switches", []))
        return f"房间[{state.room_name}] 采用[{lamp_type}]，灯具{lamp_count}个，开关{switch_count}个。"

    @staticmethod
    def _record_thought(state: RoomAgentState, action: Dict[str, Any]) -> None:
        if not isinstance(action, dict):
            return
        state.thought_history.append(
            {
                "thought": action.get("thought"),
                "action": action.get("action"),
                "args": action.get("args", {}),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _record_reflection(self, state: RoomAgentState, tool_name: str, tool_output: Dict[str, Any]) -> None:
        reflection = self.plan.reflect(tool_name, tool_output if isinstance(tool_output, dict) else {})
        item = {
            "phase": reflection.phase,
            "tool": reflection.tool,
            "ok": reflection.ok,
            "message": reflection.message,
            "next_hint": reflection.next_hint,
            "timestamp": reflection.timestamp,
        }
        state.tool_cache.setdefault("reflections", []).append(item)
        if state.logger:
            state.logger._publish("reflection", item)

    @staticmethod
    def _record_tool_error(state: RoomAgentState, tool_name: str, tool_input: Dict[str, Any], message: str) -> None:
        state.record(
            tool_name,
            tool_input,
            {"status": "error", "message": message},
            tool_result=message,
        )

    @staticmethod
    def _sync_lamp_plan(state: RoomAgentState) -> None:
        room_req = state.tool_cache.get("room_requirement", {}) or {}
        count_plan = state.tool_cache.get("component_count_plan", {}) or {}
        flux_plan = state.tool_cache.get("flux_plan", {}) or {}
        lamp_model_plan = state.tool_cache.get("lamp_model_plan", {}) or {}
        if not any([room_req, count_plan, flux_plan, lamp_model_plan]):
            return
        state.lamp_plan = {
            "room_name": state.room_name,
            "target_lux": int(room_req.get("target_lux", 300)),
            "grid_rows": int(count_plan.get("grid_rows", 1)),
            "grid_cols": int(count_plan.get("grid_cols", 1)),
            "lamp_count": int(count_plan.get("lamp_count", len(state.placements.get("lamps", [])) or 1)),
            "switch_count": int(count_plan.get("switch_count", len(state.placements.get("switches", [])))),
            "required_flux_per_lamp_lm": float(flux_plan.get("required_flux_per_lamp_lm", 0.0)),
            "selected_lamp": lamp_model_plan.get("selected_lamp", {}),
            "backup_options": lamp_model_plan.get("candidates", []),
            "spacing_m": float(count_plan.get("preferred_spacing_m", 2.4)),
            "uf": float(flux_plan.get("uf", 0.6)),
            "mf": float(flux_plan.get("mf", 0.8)),
            "is_regular": bool(count_plan.get("is_regular", True)),
        }
        state.tool_cache["lamp_plan"] = state.lamp_plan

    def _request_approval(
        self,
        approval_handler: Optional[ApprovalHandler],
        state: RoomAgentState,
        stage: str,
        snapshot_id: str,
    ) -> str:
        if approval_handler is None:
            return "continue"
        summary = {
            "lamp_count": len(state.placements.get("lamps", [])),
            "switch_count": len(state.placements.get("switches", [])),
            "selected_lamp_type": state.selected_lamp_type,
            "last_tool_result": (state.tool_cache.get("last_tool_result", {}) or {}).get("tool_result", ""),
        }
        gate = ApprovalGate(stage=stage, room_name=state.room_name, summary=summary, snapshot_id=snapshot_id)
        decision = str(approval_handler(gate, state) or "continue").strip().lower()
        if decision not in {"continue", "retry", "exit"}:
            decision = "continue"
        state.record("approval_gate", {"stage": stage, "snapshot_id": snapshot_id}, {"decision": decision})
        return decision
