from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from .logger import AgentRunLogger
from .state import RoomAgentState
from .tools import (
    MINIMAX_TOOL_AGENT_SYSTEM_PROMPT,
    REACT_TOOLS,
    LightingTools,
    build_anthropic_tools,
)


class _LegacyAnthropicLightingAgent:
    """
    使用 Anthropic 风格工具调用接口的 MiniMax 智能体。
    """

    def __init__(
        self,
        tools: Optional[LightingTools] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        log_dir: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        self.tools = tools or LightingTools()
        self.temperature = float(temperature)
        self.max_tokens = max(512, int(max_tokens))
        self.api_key = (
            os.getenv("MINIMAX_API_KEY", "").strip()
            or os.getenv("ANTHROPIC_API_KEY", "").strip()
        )
        self.base_url = (
            os.getenv("MINIMAX_BASE_URL", "").strip()
            or os.getenv("ANTHROPIC_BASE_URL", "").strip()
        )
        self.model = (
            (model_name or "").strip()
            or os.getenv("CAD_AGENT_MINIMAX_MODEL", "").strip()
            or os.getenv("MINIMAX_MODEL", "").strip()
            or "MiniMax-M2.5"
        )
        self.run_logger = AgentRunLogger(log_dir=log_dir)
        self.client: Optional[anthropic.Anthropic] = None
        if self.api_key:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url or None,
            )
        self.run_logger._emit(
            "MODEL",
            f"agent=minimax provider=anthropic-compatible model={self.model} base_url={self.base_url or 'default'}",
            AgentRunLogger.GRAY,
        )

    @staticmethod
    def list_tools() -> List[Dict[str, Any]]:
        return build_anthropic_tools(REACT_TOOLS)

    def run_for_room(
        self,
        state: RoomAgentState,
        max_steps: int = 8,
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

        if self.client is None:
            self.run_logger.error("MiniMax API key is missing, fallback to deterministic pipeline.")
            return self._run_fallback_design(state)

        latest_wiring: Optional[Dict[str, Any]] = None
        final_reason = "max_steps_reached"
        final_text = ""
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": self._build_initial_user_prompt(state, user_goal=user_goal),
            }
        ]

        for _ in range(max(1, int(max_steps))):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=MINIMAX_TOOL_AGENT_SYSTEM_PROMPT,
                    messages=messages,
                    tools=self.list_tools(),
                    temperature=self.temperature,
                )
            except Exception as exc:
                self.run_logger.error(f"minimax request failed: {exc}")
                final_reason = "llm_request_failed"
                break

            thinking_texts, text_blocks, tool_use_blocks = self._process_response_blocks(response, state)
            for thinking in thinking_texts:
                state.thought_history.append(
                    {
                        "thought": thinking,
                        "action": None,
                        "args": {},
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

            if tool_use_blocks:
                tool_results_for_model: List[Dict[str, Any]] = []
                for block in tool_use_blocks:
                    action_name = str(getattr(block, "name", "") or "")
                    action_args = dict(getattr(block, "input", {}) or {})
                    self.run_logger.action({"action": action_name, "args": action_args})
                    state.thought_history.append(
                        {
                            "thought": thinking_texts[-1] if thinking_texts else "",
                            "action": action_name,
                            "args": action_args,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    try:
                        tool_output = self.tools.invoke_tool(action_name, state, action_args)
                        self._sync_lamp_plan(state)
                        if action_name == "tool_generate_wiring":
                            latest_wiring = tool_output
                        observation_text = self._build_tool_observation(action_name, tool_output, state)
                        tool_results_for_model.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": getattr(block, "id", ""),
                                "content": observation_text,
                            }
                        )
                    except Exception as exc:
                        self.run_logger.error(f"tool execution failed: {action_name}: {exc}")
                        tool_results_for_model.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": getattr(block, "id", ""),
                                "content": f"工具执行失败：{action_name}，错误为：{exc}",
                                "is_error": True,
                            }
                        )
                messages.append({"role": "user", "content": tool_results_for_model})
                continue

            final_text = "\n".join(item for item in text_blocks if item).strip()
            final_reason = "model_final_text" if final_text else "model_no_tool_no_text"
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
            "strategy_summary": final_text or self._build_strategy_summary(state),
            "validation": final_validation,
            "stage_outputs": {
                "tool_lookup_room_requirement": state.tool_cache.get("room_requirement"),
                "tool_estimate_component_count": state.tool_cache.get("component_count_plan"),
                "tool_calc_required_flux_per_lamp": state.tool_cache.get("flux_plan"),
                "tool_retrieve_lamp_model": state.tool_cache.get("lamp_model_plan"),
                "tool_place_components": state.tool_cache.get("component_layout_plan"),
                "tool_generate_wiring": state.tool_cache.get("wiring_plan"),
            },
        }
        self.run_logger.room_end(
            room_name=state.room_name,
            finish_reason=final_reason,
            tool_calls=len(state.tool_history),
            score=(final_validation or {}).get("score"),
        )
        return result

    def _run_fallback_design(self, state: RoomAgentState) -> Dict[str, Any]:
        req = self.tools.tool_lookup_room_requirement(state)
        count_plan = self.tools.tool_estimate_component_count(state)
        flux_plan = self.tools.tool_calc_required_flux_per_lamp(
            state,
            target_lux=req.get("target_lux"),
            lamp_count=count_plan.get("lamp_count"),
        )
        model_plan = self.tools.tool_retrieve_lamp_model(
            state,
            lamp_type=req.get("lamp_type"),
            required_flux_lm=flux_plan.get("required_flux_per_lamp_lm"),
        )
        self._sync_lamp_plan(state)
        self.tools.tool_place_components(
            state,
            lamp_count=count_plan.get("lamp_count"),
            switch_count=count_plan.get("switch_count"),
            is_regular=count_plan.get("is_regular"),
        )
        validation = self.tools.tool_validate_layout(state)
        wiring = self.tools.tool_generate_wiring(state=state)
        result = {
            "room_name": state.room_name,
            "selected_lamp_type": state.selected_lamp_type,
            "lamp_plan": state.lamp_plan,
            "placements": state.placements,
            "wiring_plan": wiring,
            "log_file": self.run_logger.log_path,
            "tool_calls": len(state.tool_history),
            "finish_reason": "fallback_deterministic",
            "strategy_summary": self._build_strategy_summary(state),
            "validation": validation,
            "stage_outputs": {
                "tool_lookup_room_requirement": req,
                "tool_estimate_component_count": count_plan,
                "tool_calc_required_flux_per_lamp": flux_plan,
                "tool_retrieve_lamp_model": model_plan,
                "tool_place_components": {
                    "lamp_count": len(state.placements.get("lamps", [])),
                    "switch_count": len(state.placements.get("switches", [])),
                },
            },
        }
        self.run_logger.room_end(
            room_name=state.room_name,
            finish_reason="fallback_deterministic",
            tool_calls=len(state.tool_history),
            score=(validation or {}).get("score"),
        )
        return result

    def _process_response_blocks(
        self,
        response: Any,
        state: RoomAgentState,
    ) -> Tuple[List[str], List[str], List[Any]]:
        thinking_texts: List[str] = []
        text_blocks: List[str] = []
        tool_use_blocks: List[Any] = []

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", "")
            if block_type == "thinking":
                thinking = str(getattr(block, "thinking", "") or "").strip()
                if thinking:
                    thinking_texts.append(thinking)
                    self.run_logger.thought(thinking)
            elif block_type == "text":
                text = str(getattr(block, "text", "") or "").strip()
                if text:
                    text_blocks.append(text)
                    self.run_logger.llm_response(text)
            elif block_type == "tool_use":
                tool_use_blocks.append(block)

        if not thinking_texts and text_blocks:
            state.thought_history.append(
                {
                    "thought": text_blocks[0],
                    "action": None,
                    "args": {},
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return thinking_texts, text_blocks, tool_use_blocks

    @staticmethod
    def _build_initial_user_prompt(state: RoomAgentState, user_goal: Optional[str] = None) -> str:
        task_block = f"用户任务: {str(user_goal).strip()}\n" if user_goal else ""
        has_existing_layout = bool(state.placements.get("lamps") or state.placements.get("switches"))
        current_layout_block = (
            f"当前已有布局: 灯具{len(state.placements.get('lamps', []))}个，"
            f"开关{len(state.placements.get('switches', []))}个，"
            f"已选灯具类型{state.selected_lamp_type or '未确定'}。\n"
        )
        return (
            task_block
            + f"请为房间“{state.room_name}”完成照明设计。\n"
            + f"房间面积: {float(state.area_m2):.2f}m2\n"
            + f"矩阵尺寸: {int(state.matrix.shape[0])}x{int(state.matrix.shape[1])}\n"
            + f"{current_layout_block}"
            + f"{'请根据用户任务调整现有布局。' if has_existing_layout else '当前布局为空，请先根据房间需求完成灯具选型、数量估算和元件布置。'}\n"
            + "如果你需要查看网格细节，请调用 tool_read_matrix_state。\n"
        )

    @staticmethod
    def _build_strategy_summary(state: RoomAgentState) -> str:
        lamp_type = state.selected_lamp_type or "未知灯具"
        lamp_count = len(state.placements.get("lamps", []))
        switch_count = len(state.placements.get("switches", []))
        return f"房间[{state.room_name}] 采用[{lamp_type}]，灯具{lamp_count}个，开关{switch_count}个。"

    @staticmethod
    def _build_tool_observation(
        tool_name: str,
        tool_output: Dict[str, Any],
        state: RoomAgentState,
    ) -> str:
        last_tool_result = (
            (state.tool_cache.get("last_tool_result", {}) or {}).get("tool_result", "")
            if isinstance(state.tool_cache, dict)
            else ""
        )
        parts: List[str] = [last_tool_result or f"{tool_name} 已执行。"]

        if tool_name == "tool_read_matrix_state":
            board = str(tool_output.get("ascii_board", "") or "").strip()
            if board:
                parts.append("ASCII棋盘如下：")
                parts.append(board)
        elif tool_name == "tool_validate_layout":
            violations = tool_output.get("violations", []) or []
            suggestions = tool_output.get("suggestions", []) or []
            if violations:
                parts.append(
                    "违规详情："
                    + "；".join(
                        f"{item.get('code', 'UNKNOWN')}:{item.get('message', '')}"
                        for item in violations[:4]
                        if isinstance(item, dict)
                    )
                )
            if suggestions:
                parts.append("修正建议：" + "；".join(str(item) for item in suggestions[:3]))
        elif tool_name == "tool_place_components":
            parts.append(
                f"灯具坐标为{tool_output.get('lamp_positions', [])}，"
                f"开关坐标为{tool_output.get('switch_positions', [])}。"
            )
        elif tool_name == "tool_generate_wiring":
            parts.append(
                f"已生成{tool_output.get('route_count', 0)}条布线路径，"
                f"合并后线段数为{len(tool_output.get('merged_segments_grid', []) or [])}。"
            )

        return "\n".join(part for part in parts if part)

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
