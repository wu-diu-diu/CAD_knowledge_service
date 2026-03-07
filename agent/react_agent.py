from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .logger import AgentRunLogger
from .state import RoomAgentState
from .tools import LightingTools, REACT_SYSTEM_PROMPT_TEMPLATE, REACT_TOOLS
from .utils import describe_tool_result, extract_json, resolve_provider


class ReActLightingAgent:
    """
    ReAct 智能体:
    - `init_mode=rule` 时先用规则生成初稿
    - `init_mode=llm` 时从空布局开始由模型逐步调用工具
    - 每轮通过 thought/action/tool_result 循环完成布局与布线
    """

    def __init__(
        self,
        tools: Optional[LightingTools] = None,
        provider: str = "qwen",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        init_mode: str = "llm",
        log_dir: Optional[str] = None,
    ) -> None:
        self.tools = tools or LightingTools()
        self.provider = provider.strip().lower()
        self.temperature = float(temperature)
        self.init_mode = (init_mode or "rule").strip().lower()
        if self.init_mode not in ("rule", "llm"):
            self.init_mode = "rule"
        self.api_key, self.base_url, self.model = resolve_provider(self.provider, model_name)
        self.run_logger = AgentRunLogger(log_dir=log_dir)
        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
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
        max_steps: int = 6,
        user_goal: Optional[str] = None,
        reset_layout: bool = True,
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

        if self.init_mode == "rule":
            init_stage = self._run_replan_design(state, args={})
            first_validation = init_stage["validation"]
        else:
            first_validation = self.tools.tool_validate_layout(state)
            init_stage = {
                "req": None,
                "count_plan": None,
                "flux_plan": None,
                "model_plan": None,
                "coords_plan": None,
                "validation": first_validation,
            }
            state.record("init_mode", {"mode": "llm"}, {"status": "start_from_empty"})

        latest_wiring: Optional[Dict[str, Any]] = None
        available_tools = self.list_tools()
        valid_actions = {"finish"} | {
            str(item.get("function", {}).get("name", ""))
            for item in available_tools
            if isinstance(item, dict)
        }

        final_reason = "max_steps_reached"
        last_tool_output: Optional[Dict[str, Any]] = state.tool_cache.get("last_tool_result")
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

            if isinstance(action, dict):
                state.thought_history.append(
                    {
                        "thought": action.get("thought"),
                        "action": action.get("action"),
                        "args": action.get("args", {}),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            action_name = action.get("action")
            if action_name == "finish":
                final_reason = str(action.get("reason", "done"))
                state.record("finish", {}, {"reason": final_reason, "strategy": action.get("strategy", "")})
                last_tool_output = state.tool_cache.get("last_tool_result")
                break

            if action_name == "tool_validate_layout":
                args = action.get("args", {}) or {}
                self.tools.tool_validate_layout(
                    state=state,
                    min_lamp_dist_cells=int(args.get("min_lamp_dist_cells", 2)),
                    max_switch_to_door_dist_cells=int(args.get("max_switch_to_door_dist_cells", 3)),
                )
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name == "tool_lookup_room_requirement":
                args = action.get("args", {}) or {}
                self.tools.tool_lookup_room_requirement(state=state, room_name=args.get("room_name"))
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name == "tool_estimate_component_count":
                args = action.get("args", {}) or {}
                self.tools.tool_estimate_component_count(
                    state=state,
                    is_regular=args.get("is_regular"),
                    min_spacing_m=float(args.get("min_spacing_m", 2.0)),
                    max_spacing_m=float(args.get("max_spacing_m", 3.0)),
                    max_lamps=int(args.get("max_lamps", 64)),
                    switch_count=args.get("switch_count"),
                )
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name == "tool_calc_required_flux_per_lamp":
                args = action.get("args", {}) or {}
                self.tools.tool_calc_required_flux_per_lamp(
                    state=state,
                    target_lux=args.get("target_lux"),
                    lamp_count=args.get("lamp_count"),
                    uf=float(args.get("uf", 0.6)),
                    mf=float(args.get("mf", 0.8)),
                )
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name == "tool_retrieve_lamp_model":
                args = action.get("args", {}) or {}
                self.tools.tool_retrieve_lamp_model(
                    state=state,
                    lamp_type=args.get("lamp_type"),
                    required_flux_lm=args.get("required_flux_lm"),
                    top_k=int(args.get("top_k", 3)),
                )
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name == "tool_place_components":
                args = action.get("args", {}) or {}
                self.tools.tool_place_components(
                    state=state,
                    lamp_count=args.get("lamp_count"),
                    switch_count=args.get("switch_count"),
                    is_regular=args.get("is_regular"),
                )
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name == "tool_read_matrix_state":
                args = action.get("args", {}) or {}
                self.tools.tool_read_matrix_state(
                    state=state,
                    max_rows=int(args.get("max_rows", 64)),
                    max_cols=int(args.get("max_cols", 64)),
                    compress=bool(args.get("compress", True)),
                )
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name == "tool_generate_wiring":
                args = action.get("args", {}) or {}
                latest_wiring = self.tools.tool_generate_wiring(
                    state=state,
                    turn_penalty=args.get("turn_penalty"),
                    bbox_pixel=args.get("bbox_pixel"),
                    cell_size_px=int(args.get("cell_size_px", 40)),
                    cad_params=args.get("cad_params"),
                    image_width=args.get("image_width"),
                    image_height=args.get("image_height"),
                )
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name == "tool_apply_layout_edit":
                args = action.get("args", {}) or {}
                self.tools.tool_apply_layout_edit(state=state, edits=args.get("edits", []))
                last_tool_output = state.tool_cache.get("last_tool_result")
                continue

            if action_name not in valid_actions:
                final_reason = f"unknown_action:{action_name}"
                break
            final_reason = f"unhandled_action:{action_name}"
            break

        final_validation = self.tools.tool_validate_layout(state)
        if latest_wiring is None:
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
                "tool_lookup_room_requirement": init_stage["req"],
                "tool_estimate_component_count": init_stage["count_plan"],
                "tool_calc_required_flux_per_lamp": init_stage["flux_plan"],
                "tool_retrieve_lamp_model": init_stage["model_plan"],
                "tool_place_components": init_stage["coords_plan"],
                "initial_validation": first_validation,
            },
        }
        self.run_logger.room_end(
            room_name=state.room_name,
            finish_reason=final_reason,
            tool_calls=len(state.tool_history),
            score=(final_validation or {}).get("score"),
        )
        return result

    def _run_replan_design(
        self,
        state: RoomAgentState,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
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
        if args.get("lamp_count") is not None:
            lamp_count = max(1, int(args["lamp_count"]))
            count_plan["lamp_count"] = lamp_count
        else:
            lamp_count = int(count_plan.get("lamp_count", 1))

        if args.get("switch_count") is not None:
            component_switch_count = max(0, int(args["switch_count"]))
            count_plan["switch_count"] = component_switch_count
        else:
            component_switch_count = int(count_plan.get("switch_count", 1))

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
        last_tool_output: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        user_goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            return {"action": "finish", "reason": "no_llm_key", "strategy": "deterministic tools only"}

        system_prompt = self._build_system_prompt(tools or self.list_tools())
        user_prompt = self._build_user_prompt(state, view, validation, last_tool_output, user_goal=user_goal)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            content = ((resp.choices or [{}])[0].message.content or "").strip()
            if state.logger:
                state.logger.llm_response(content)
            obj = extract_json(content)
            if isinstance(obj, dict) and obj.get("action"):
                return obj
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
        prompt_parts = []

        if user_goal:
            prompt_parts.extend(
                [
                    "=== 用户任务 ===",
                    str(user_goal).strip(),
                    "",
                ]
            )

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
                action_str = f"action={item.get('action')}"
                prompt_parts.append(f"{idx}. {str(item.get('thought', ''))[:80]} [{action_str}]")

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
