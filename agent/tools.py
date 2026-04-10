from __future__ import annotations

import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from preprocess.lighting_layout import (
    _select_lamp_cells_regular_grid,
    _select_lamp_cells_rule_based,
)
from preprocess.wiring_layout import (
    _build_edge_candidates,
    _build_mst,
    _grid_path_to_pixel_path,
    _merge_unique_step_segments,
    _orient_edges_from_switch,
    _pixel_path_to_cad_path,
)

from .state import RoomAgentState
from .utils import (
    count_binary_components,
    get_target_lux,
    is_point,
    parse_flux_lm,
    parse_power_w,
    search_grid_shape,
)


@dataclass
class LampSpec:
    lamp_type: str
    model: str
    flux_lm: float
    power_w: Optional[float] = None
    vendor: str = ""
    url: str = ""


def _build_lamp_specs(raw_catalog: List[Dict[str, Any]]) -> List[LampSpec]:
    specs: List[LampSpec] = []
    for item in raw_catalog:
        lamp_type = str(item.get("灯具类型", "")).strip()
        if not lamp_type:
            continue
        specs.append(
            LampSpec(
                lamp_type=lamp_type,
                model=str(item.get("型号", "")).strip(),
                flux_lm=parse_flux_lm(item.get("光通量", "")),
                power_w=parse_power_w(item.get("功率", "")),
                vendor=str(item.get("厂家", "")).strip(),
                url=str(item.get("购买链接", "")).strip(),
            )
        )
    return specs


def _lamp_to_dict(spec: LampSpec) -> Dict[str, Any]:
    return {
        "lamp_type": spec.lamp_type,
        "model": spec.model,
        "flux_lm": spec.flux_lm,
        "power_w": spec.power_w,
        "vendor": spec.vendor,
        "url": spec.url,
    }


def _normalize_cad_params(cad_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not isinstance(cad_params, dict):
        return None

    candidate: Dict[str, Any] = cad_params
    for nested_key in ("cad_params", "default_cad_params"):
        nested = candidate.get(nested_key)
        if isinstance(nested, dict):
            candidate = nested
            break

    alias_map = {
        "Xmin": ("Xmin", "xmin", "x_min", "xMin"),
        "Ymin": ("Ymin", "ymin", "y_min", "yMin"),
        "Xmax": ("Xmax", "xmax", "x_max", "xMax"),
        "Ymax": ("Ymax", "ymax", "y_max", "yMax"),
    }
    resolved: Dict[str, float] = {}
    for canonical_key, aliases in alias_map.items():
        value = None
        for alias in aliases:
            if candidate.get(alias) is not None:
                value = candidate.get(alias)
                break
        if value is None:
            return None
        resolved[canonical_key] = float(value)
    return resolved


class LightingTools:
    def __init__(
        self,
        lamp_catalog: Optional[List[Dict[str, Any]]] = None,
        room_lux_requirements: Optional[Dict[str, int]] = None,
    ) -> None:
        self.catalog = _build_lamp_specs(lamp_catalog or DEFAULT_LAMP_CATALOG)
        self.room_lux_map = dict(DEFAULT_ROOM_LUX)
        if room_lux_requirements:
            self.room_lux_map.update(room_lux_requirements)

    def infer_default_lamp_type(self, room_name: str) -> Dict[str, Any]:
        normalized_name = (room_name or "").strip()
        target_lux = get_target_lux(normalized_name, self.room_lux_map)

        lamp_type = "筒灯"
        if any(key in normalized_name for key in ("配电", "除尘", "高温")):
            lamp_type = "防爆灯"
        elif any(key in normalized_name for key in ("楼梯", "卫生间", "盟洗")):
            lamp_type = "感应式吸顶灯"
        elif target_lux >= 500:
            lamp_type = "双管格栅灯"
        elif target_lux >= 300:
            lamp_type = "双管荧光灯"

        return {
            "room_name": normalized_name,
            "target_lux": int(target_lux),
            "lamp_type": lamp_type,
        }

    def tool_lookup_room_requirement(
        self,
        state: RoomAgentState,
        room_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_name = (room_name or state.room_name or "").strip()
        inferred = self.infer_default_lamp_type(normalized_name)
        preferred_lamp = str(inferred.get("lamp_type", "筒灯"))
        target_lux = int(inferred.get("target_lux", 300))
        result = {
            "room_name": normalized_name,
            "target_lux": target_lux,
            "preferred_lamp_types": [preferred_lamp],
            "lamp_type": preferred_lamp,
        }
        state.selected_lamp_type = preferred_lamp
        state.tool_cache["room_requirement"] = result
        state.record("tool_lookup_room_requirement", {"room_name": normalized_name}, result)
        return result

    def tool_estimate_component_count(
        self,
        state: RoomAgentState,
        is_regular: Optional[bool] = None,
        min_spacing_m: float = 2.0,
        max_spacing_m: float = 3.0,
        max_lamps: int = 64,
        switch_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        valid = np.argwhere(state.matrix > 0)
        if len(valid) == 0:
            valid = np.argwhere(np.ones_like(state.matrix, dtype=np.int32) > 0)

        r_min, c_min = valid.min(axis=0)
        r_max, c_max = valid.max(axis=0)
        h_cells = int(r_max - r_min + 1)
        w_cells = int(c_max - c_min + 1)
        bbox_area_cells = max(1, h_cells * w_cells)
        fill_ratio = float(len(valid)) / float(bbox_area_cells)

        if is_regular is None:
            is_regular = bool(fill_ratio >= 0.90)

        aspect = float(w_cells) / max(1.0, float(h_cells))
        area_m2 = max(0.01, float(state.area_m2))
        room_w_m = max(0.1, math.sqrt(area_m2 * max(aspect, 1e-3)))
        room_h_m = max(0.1, area_m2 / room_w_m)

        base_spacing = (max(0.5, float(min_spacing_m)) + max(0.5, float(max_spacing_m))) / 2.0
        preferred_spacing = base_spacing if is_regular else max(0.8, base_spacing * 0.85)
        rough_count = int(math.ceil(area_m2 / max(0.25, preferred_spacing * preferred_spacing)))
        rough_count = max(1, min(int(max_lamps), rough_count))

        grid_rows, grid_cols = search_grid_shape(
            target_count=rough_count,
            aspect=aspect,
            room_w_m=room_w_m,
            room_h_m=room_h_m,
            preferred_spacing_m=preferred_spacing,
        )
        lamp_count = int(grid_rows * grid_cols)

        if switch_count is None:
            switch_count_val = int(count_binary_components(state.matrix == 2))
        else:
            switch_count_val = max(0, int(switch_count))

        result = {
            "room_name": state.room_name,
            "is_regular": bool(is_regular),
            "fill_ratio": float(fill_ratio),
            "aspect_ratio": float(aspect),
            "grid_rows": int(grid_rows),
            "grid_cols": int(grid_cols),
            "lamp_count": int(lamp_count),
            "switch_count": int(switch_count_val),
            "component_count": {
                "lamps": int(lamp_count),
                "switches": int(switch_count_val),
            },
            "preferred_spacing_m": float(preferred_spacing),
            "room_w_m": float(room_w_m),
            "room_h_m": float(room_h_m),
        }
        state.tool_cache["component_count_plan"] = result
        state.record(
            "tool_estimate_component_count",
            {
                "is_regular": is_regular,
                "min_spacing_m": min_spacing_m,
                "max_spacing_m": max_spacing_m,
                "max_lamps": max_lamps,
                "switch_count": switch_count,
            },
            result,
        )
        return result

    def tool_calc_required_flux_per_lamp(
        self,
        state: RoomAgentState,
        target_lux: Optional[int] = None,
        lamp_count: Optional[int] = None,
        uf: float = 0.6,
        mf: float = 0.8,
    ) -> Dict[str, Any]:
        """
        照度计算公式：
        E = (N * F * UF * MF) / A
        其中：
        E = 照度 (lux)
        N = 灯具数量
        F = 每盏灯具的光通量 (lumen)
        UF = 利用系数 (0~1，考虑灯具设计和安装对光的利用效率)
        MF = 维护系数 (0~1，考虑灯具和环境的清洁程度对光输出的影响)
        A = 房间面积 (平方米)
        """
        cached_req = state.tool_cache.get("room_requirement", {}) or {}
        cached_count = state.tool_cache.get("component_count_plan", {}) or {}
        resolved_target_lux = int(target_lux if target_lux is not None else cached_req.get("target_lux", 300))
        resolved_lamp_count = int(lamp_count if lamp_count is not None else cached_count.get("lamp_count", 1))

        count = max(1, int(resolved_lamp_count))
        area_m2 = max(0.01, float(state.area_m2))
        util = max(1e-6, float(uf) * float(mf))
        required_flux = (float(resolved_target_lux) * area_m2) / max(1e-6, util * count)
        result = {
            "room_name": state.room_name,
            "target_lux": int(resolved_target_lux),
            "area_m2": float(area_m2),
            "lamp_count": int(count),
            "uf": float(uf),
            "mf": float(mf),
            "required_flux_per_lamp_lm": float(required_flux),
        }
        state.tool_cache["flux_plan"] = result
        state.record(
            "tool_calc_required_flux_per_lamp",
            {
                "target_lux": target_lux,
                "lamp_count": lamp_count,
                "resolved_target_lux": resolved_target_lux,
                "resolved_lamp_count": resolved_lamp_count,
                "uf": uf,
                "mf": mf,
            },
            result,
        )
        return result

    def tool_retrieve_lamp_model(
        self,
        state: RoomAgentState,
        lamp_type: Optional[str] = None,
        required_flux_lm: Optional[float] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        cached_flux = state.tool_cache.get("flux_plan", {}) or {}
        normalized_type = (lamp_type or state.selected_lamp_type or "筒灯").strip()
        resolved_flux = float(
            required_flux_lm
            if required_flux_lm is not None
            else cached_flux.get("required_flux_per_lamp_lm", 1000.0)
        )
        candidates = [item for item in self.catalog if item.lamp_type == normalized_type]
        match_scope = "by_type"
        if not candidates:
            candidates = list(self.catalog)
            match_scope = "fallback_all_types"
        if not candidates:
            raise ValueError("lamp catalog is empty")

        ranked = sorted(candidates, key=lambda item: abs(float(item.flux_lm) - float(resolved_flux)))
        selected = ranked[0]
        top = ranked[: max(1, int(top_k))]
        result = {
            "room_name": state.room_name,
            "requested_lamp_type": normalized_type,
            "required_flux_lm": float(resolved_flux),
            "selected_lamp": _lamp_to_dict(selected),
            "candidates": [_lamp_to_dict(item) for item in top],
            "match_scope": match_scope,
        }
        state.selected_lamp_type = str(result["selected_lamp"].get("lamp_type", normalized_type))
        state.tool_cache["lamp_model_plan"] = result
        state.record(
            "tool_retrieve_lamp_model",
            {
                "lamp_type": lamp_type,
                "required_flux_lm": required_flux_lm,
                "resolved_lamp_type": normalized_type,
                "resolved_required_flux_lm": resolved_flux,
                "top_k": top_k,
            },
            result,
        )
        return result

    def tool_place_components(
        self,
        state: RoomAgentState,
        lamp_count: Optional[int] = None,
        switch_count: Optional[int] = None,
        is_regular: Optional[bool] = None,
    ) -> Dict[str, Any]:
        planned_count = int(
            lamp_count
            if lamp_count is not None
            else ((state.lamp_plan or {}).get("lamp_count", 0) or len(state.placements.get("lamps", [])) or 1)
        )
        planned_count = max(1, planned_count)
        cached_component_plan = state.tool_cache.get("component_count_plan", {}) or {}
        planned_switch_count = int(
            switch_count
            if switch_count is not None
            else (
                (state.lamp_plan or {}).get("switch_count", None)
                if (state.lamp_plan or {}).get("switch_count", None) is not None
                else cached_component_plan.get("switch_count", 1)
            )
        )
        planned_switch_count = max(0, planned_switch_count)

        if is_regular is None:
            valid = np.argwhere(state.matrix > 0)
            if len(valid) == 0:
                fill_ratio = 1.0
            else:
                r_min, c_min = valid.min(axis=0)
                r_max, c_max = valid.max(axis=0)
                bbox_area = max(1, int(r_max - r_min + 1) * int(c_max - c_min + 1))
                fill_ratio = float(len(valid)) / float(bbox_area)
            is_regular = bool(fill_ratio >= 0.90)

        if bool(is_regular):
            picked = _select_lamp_cells_regular_grid(state.matrix, planned_count)
            algo = "step7_regular_grid"
        else:
            picked = _select_lamp_cells_rule_based(state.matrix, planned_count)
            algo = "step7_irregular_rule"

        state.placements["lamps"] = [[int(row), int(col)] for row, col in picked]
        rows, cols = state.matrix.shape
        door_cells = [tuple(map(int, point)) for point in np.argwhere(state.matrix == 2).tolist()]
        valid_cells = [tuple(map(int, point)) for point in np.argwhere(state.matrix == 1).tolist()]
        switch_candidates: List[Tuple[int, int]] = []
        if door_cells:
            for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for row, col in door_cells:
                    next_row, next_col = row + delta_row, col + delta_col
                    if 0 <= next_row < rows and 0 <= next_col < cols and int(state.matrix[next_row, next_col]) == 1:
                        switch_candidates.append((next_row, next_col))
        if not switch_candidates:
            switch_candidates = valid_cells

        unique_candidates: List[Tuple[int, int]] = []
        for item in switch_candidates:
            if item not in unique_candidates:
                unique_candidates.append(item)

        if valid_cells:
            center_row = float(np.mean([item[0] for item in valid_cells]))
            center_col = float(np.mean([item[1] for item in valid_cells]))
            unique_candidates = sorted(
                unique_candidates,
                key=lambda item: (
                    min((abs(item[0] - door[0]) + abs(item[1] - door[1])) for door in door_cells) if door_cells else 0,
                    abs(item[0] - center_row) + abs(item[1] - center_col),
                ),
            )
        picked_switches = unique_candidates[:planned_switch_count]
        state.placements["switches"] = [[int(row), int(col)] for row, col in picked_switches]

        result = {
            "component_type": "components",
            "lamp_positions": state.placements["lamps"],
            "switch_positions": state.placements["switches"],
            "lamp_count": len(state.placements["lamps"]),
            "switch_count": len(state.placements["switches"]),
            "requested_lamp_count": int(planned_count),
            "requested_switch_count": int(planned_switch_count),
            "is_regular": bool(is_regular),
            "lamp_algorithm": algo,
        }
        state.tool_cache["component_layout_plan"] = result
        state.record(
            "tool_place_components",
            {
                "lamp_count": lamp_count,
                "switch_count": switch_count,
                "is_regular": is_regular,
            },
            result,
        )
        return result

    def tool_validate_layout(
        self,
        state: RoomAgentState,
        min_lamp_dist_cells: int = 2,
        max_switch_to_door_dist_cells: int = 3,
    ) -> Dict[str, Any]:
        rows, cols = state.matrix.shape
        lamps = [tuple(map(int, item)) for item in state.placements.get("lamps", []) if len(item) == 2]
        switches = [tuple(map(int, item)) for item in state.placements.get("switches", []) if len(item) == 2]
        door_cells = [tuple(map(int, item)) for item in np.argwhere(state.matrix == 2).tolist()]

        violations: List[Dict[str, Any]] = []

        def add_violation(code: str, severity: str, message: str) -> None:
            violations.append({"code": code, "severity": severity, "message": message})

        if len(lamps) == 0:
            add_violation("NO_LAMP", "critical", "未放置任何灯具")
        if len(switches) == 0:
            add_violation("NO_SWITCH", "warning", "未放置开关")

        for row, col in lamps:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                add_violation("LAMP_OUT_OF_RANGE", "critical", f"灯具坐标越界: [{row},{col}]")
            elif int(state.matrix[row, col]) != 1:
                add_violation("LAMP_ON_INVALID_CELL", "critical", f"灯具未落在可布置格: [{row},{col}]")

        for row, col in switches:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                add_violation("SWITCH_OUT_OF_RANGE", "critical", f"开关坐标越界: [{row},{col}]")
            elif int(state.matrix[row, col]) != 1:
                add_violation("SWITCH_ON_INVALID_CELL", "critical", f"开关未落在可布置格: [{row},{col}]")

        if len(set(lamps)) != len(lamps):
            add_violation("LAMP_DUPLICATE_POSITION", "warning", "存在灯具坐标重叠")
        if len(set(switches)) != len(switches):
            add_violation("SWITCH_DUPLICATE_POSITION", "warning", "存在开关坐标重叠")

        min_lamp_dist = None
        if len(lamps) >= 2:
            min_lamp_dist = 10**9
            for idx in range(len(lamps)):
                for jdx in range(idx + 1, len(lamps)):
                    dist = math.sqrt((lamps[idx][0] - lamps[jdx][0]) ** 2 + (lamps[idx][1] - lamps[jdx][1]) ** 2)
                    min_lamp_dist = min(min_lamp_dist, dist)
            if min_lamp_dist < float(min_lamp_dist_cells):
                add_violation("LAMPS_TOO_CLOSE", "warning", f"部分灯具间距过近: min={min_lamp_dist:.2f}格")

        coverage_cv = None
        valid_cells = np.argwhere(state.matrix == 1)
        if len(valid_cells) > 0 and len(lamps) > 0:
            distances: List[float] = []
            for row, col in valid_cells.tolist():
                best = min(math.sqrt((row - lamp_row) ** 2 + (col - lamp_col) ** 2) for lamp_row, lamp_col in lamps)
                distances.append(float(best))
            if distances:
                mean_dist = float(np.mean(distances))
                std_dist = float(np.std(distances))
                coverage_cv = std_dist / max(1e-6, mean_dist)
                if coverage_cv > 0.65:
                    add_violation("UNEVEN_COVERAGE", "warning", f"照明分布不均匀: cv={coverage_cv:.2f}")

        switch_to_door = None
        if switches and door_cells:
            min_dist = min(abs(s_row - d_row) + abs(s_col - d_col) for s_row, s_col in switches for d_row, d_col in door_cells)
            switch_to_door = float(min_dist)
            if min_dist > int(max_switch_to_door_dist_cells):
                add_violation("SWITCH_FAR_FROM_DOOR", "warning", f"开关距门过远: min_manhattan={min_dist}")

        score = 100
        for item in violations:
            score -= 30 if item["severity"] == "critical" else 10
        score = max(0, int(score))
        critical_count = sum(1 for item in violations if item["severity"] == "critical")
        is_valid = bool(critical_count == 0 and score >= 70)

        suggestions: List[str] = []
        codes = {item["code"] for item in violations}
        if "NO_SWITCH" in codes or "SWITCH_FAR_FROM_DOOR" in codes:
            suggestions.append("将开关移动到靠近门格(D)的一侧边缘可布置格")
        if "LAMPS_TOO_CLOSE" in codes or "UNEVEN_COVERAGE" in codes:
            suggestions.append("拉大灯具间距并向房间几何中心对称分布")
        if "LAMP_ON_INVALID_CELL" in codes or "SWITCH_ON_INVALID_CELL" in codes:
            suggestions.append("将元件移动到值为1的可布置网格")

        result = {
            "score": score,
            "is_valid": is_valid,
            "violations": violations,
            "stats": {
                "lamp_count": len(lamps),
                "switch_count": len(switches),
                "min_lamp_dist_cells": None if min_lamp_dist is None else float(min_lamp_dist),
                "coverage_cv": None if coverage_cv is None else float(coverage_cv),
                "switch_to_door_manhattan": switch_to_door,
            },
            "suggestions": suggestions,
        }
        state.record(
            "tool_validate_layout",
            {
                "min_lamp_dist_cells": min_lamp_dist_cells,
                "max_switch_to_door_dist_cells": max_switch_to_door_dist_cells,
            },
            result,
        )
        return result

    def tool_apply_layout_edit(
        self,
        state: RoomAgentState,
        edits: Any,
    ) -> Dict[str, Any]:
        if isinstance(edits, dict):
            edit_list = [edits]
        elif isinstance(edits, list):
            edit_list = edits
        else:
            edit_list = []

        applied = 0
        errors: List[str] = []
        for idx, edit in enumerate(edit_list):
            comp = str((edit or {}).get("component_type", "lamps"))
            source = (edit or {}).get("source")
            target = (edit or {}).get("target")
            ok, err = self._apply_single_move(state=state, component_type=comp, source=source, target=target)
            if ok:
                applied += 1
            else:
                errors.append(f"edit[{idx}]: {err}")

        result = {
            "ok": len(errors) == 0,
            "applied": int(applied),
            "errors": errors,
            "placements": state.placements,
        }
        state.record("tool_apply_layout_edit", {"edits": edit_list}, result)
        return result

    def _apply_single_move(
        self,
        state: RoomAgentState,
        component_type: str,
        source: Optional[List[int]],
        target: Any,
    ) -> Tuple[bool, Optional[str]]:
        rows, cols = state.matrix.shape
        if component_type not in ("lamps", "switches"):
            return False, f"invalid component_type: {component_type}"
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return False, "invalid target: must be [row, col]"

        target_row, target_col = int(target[0]), int(target[1])
        if target_row < 0 or target_row >= rows or target_col < 0 or target_col >= cols:
            return False, "target out of range"
        if int(state.matrix[target_row, target_col]) != 1:
            return False, "target cell is not placeable (must be 1)"

        placements = state.placements.setdefault(component_type, [])
        source_pair = [int(source[0]), int(source[1])] if source and len(source) == 2 else None
        target_pair = [target_row, target_col]

        if source_pair and source_pair in placements:
            idx = placements.index(source_pair)
            placements[idx] = target_pair
        else:
            placements.append(target_pair)
        return True, None

    def tool_generate_wiring(
        self,
        state: RoomAgentState,
        turn_penalty: Optional[float] = None,
        bbox_pixel: Optional[List[int]] = None,
        cell_size_px: int = 40,
        cad_params: Optional[Dict[str, float]] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        matrix = np.array(state.matrix, dtype=np.int32)
        resolved_cad_params = _normalize_cad_params(cad_params)
        lamp_points: List[Tuple[int, int]] = []
        switch_points: List[Tuple[int, int]] = []

        for item in state.placements.get("lamps", []):
            if is_point(item):
                lamp_points.append((int(item[0]), int(item[1])))
        for item in state.placements.get("switches", []):
            if is_point(item):
                switch_points.append((int(item[0]), int(item[1])))

        if not lamp_points or not switch_points:
            result = {
                "room_name": state.room_name,
                "status": "skipped",
                "reason": "missing_switch_or_lamps",
                "route_count": 0,
                "routes": [],
            }
            state.record(
                "tool_generate_wiring",
                {
                    "turn_penalty": turn_penalty,
                    "bbox_pixel": bbox_pixel,
                    "cell_size_px": cell_size_px,
                    "has_cad": bool(resolved_cad_params),
                    "image_width": image_width,
                    "image_height": image_height,
                },
                result,
            )
            return result

        penalty = (
            float(turn_penalty)
            if turn_penalty is not None
            else float(os.getenv("CAD_WIRING_TURN_PENALTY", "0.8"))
        )
        switch_point = switch_points[0]
        nodes: List[Tuple[int, int]] = [switch_point] + lamp_points
        node_labels = ["switch"] + [f"lamp_{idx + 1}" for idx in range(len(lamp_points))]

        edge_candidates = _build_edge_candidates(matrix, nodes, turn_penalty=penalty)
        mst_edges = _build_mst(edge_candidates, len(nodes))
        directed = _orient_edges_from_switch(mst_edges, nodes, switch_idx=0)

        can_export_pixel = bool(bbox_pixel and len(bbox_pixel) == 4)
        can_export_cad = bool(can_export_pixel and resolved_cad_params and image_width and image_height)

        routes: List[Dict[str, Any]] = []
        route_paths_grid: List[List[Tuple[int, int]]] = []
        total_cost = 0.0
        for parent_idx, child_idx, path_grid, cost in directed:
            route_paths_grid.append(path_grid)
            total_cost += float(cost)

            route_obj: Dict[str, Any] = {
                "from_node": node_labels[parent_idx],
                "to_node": node_labels[child_idx],
                "from_grid": [int(nodes[parent_idx][0]), int(nodes[parent_idx][1])],
                "to_grid": [int(nodes[child_idx][0]), int(nodes[child_idx][1])],
                "path_grid": [[int(row), int(col)] for row, col in path_grid],
                "cost": float(cost),
            }
            if can_export_pixel:
                pixel_path = _grid_path_to_pixel_path(path_grid, bbox_pixel, int(cell_size_px))
                route_obj["path_pixel"] = pixel_path
                if can_export_cad:
                    cad_path = _pixel_path_to_cad_path(
                        pixel_path=pixel_path,
                        cad_params=resolved_cad_params or {},
                        image_w=int(image_width),
                        image_h=int(image_height),
                    )
                    route_obj["path_cad"] = cad_path
            routes.append(route_obj)

        unreachable_nodes: List[str] = []
        reachable = {0}
        for _, child_idx, _, _ in directed:
            reachable.add(child_idx)
        for idx in range(1, len(nodes)):
            if idx not in reachable:
                unreachable_nodes.append(node_labels[idx])

        merged_segments_grid = _merge_unique_step_segments(route_paths_grid)
        merged_segments_pixel: List[List[List[float]]] = []
        merged_segments_cad: List[List[List[float]]] = []
        if can_export_pixel:
            for segment in merged_segments_grid:
                point0, point1 = tuple(segment[0]), tuple(segment[1])
                pixel_segment = _grid_path_to_pixel_path([point0, point1], bbox_pixel, int(cell_size_px))
                merged_segments_pixel.append(pixel_segment)
                if can_export_cad:
                    cad_segment = _pixel_path_to_cad_path(
                        pixel_path=pixel_segment,
                        cad_params=resolved_cad_params or {},
                        image_w=int(image_width),
                        image_h=int(image_height),
                    )
                    merged_segments_cad.append(cad_segment)

        status = "ok"
        if len(directed) < max(0, len(nodes) - 1):
            status = "partial"

        result = {
            "room_name": state.room_name,
            "status": status,
            "turn_penalty": float(penalty),
            "node_count": len(nodes),
            "route_count": len(routes),
            "switch_grid_position": [int(switch_point[0]), int(switch_point[1])],
            "lamp_grid_positions": [[int(row), int(col)] for row, col in lamp_points],
            "total_route_cost": float(total_cost),
            "unreachable_nodes": unreachable_nodes,
            "routes": routes,
            "merged_segments_grid": merged_segments_grid,
            "merged_segments_pixel": merged_segments_pixel,
            "merged_segments_cad": merged_segments_cad,
        }
        state.tool_cache["wiring_plan"] = result
        state.record(
            "tool_generate_wiring",
            {
                "turn_penalty": penalty,
                "bbox_pixel": bbox_pixel,
                "cell_size_px": cell_size_px,
                "has_cad": bool(resolved_cad_params),
                "image_width": image_width,
                "image_height": image_height,
            },
            {
                "status": status,
                "route_count": len(routes),
                "unreachable_nodes": len(unreachable_nodes),
            },
        )
        return result

    def tool_parse_user_requirement(
        self,
        state: RoomAgentState,
        user_input: str = "",
    ) -> Dict[str, Any]:
        """解析用户自然语言需求，提取结构化设计约束。"""
        text = (user_input or "").strip()

        # 色温关键词
        color_temp = None
        if any(k in text for k in ("暖光", "暖白", "暖色", "温馨")):
            color_temp = "暖白(2700-3500K)"
        elif any(k in text for k in ("冷光", "冷白", "冷色", "白光")):
            color_temp = "冷白(5000-6500K)"
        elif any(k in text for k in ("中性", "自然光", "中白")):
            color_temp = "中性白(3500-5000K)"

        # 照度偏好
        lux_override = None
        lux_match = re.search(r"(\d{2,4})\s*(?:lux|lx|勒克斯)", text, re.IGNORECASE)
        if lux_match:
            lux_override = int(lux_match.group(1))

        # 灯具类型偏好
        lamp_type_pref = None
        for lamp_kw in ("筒灯", "吸顶灯", "格栅灯", "荧光灯", "防爆灯", "感应"):
            if lamp_kw in text:
                lamp_type_pref = lamp_kw
                break

        # 节能/预算偏好
        energy_saving = any(k in text for k in ("节能", "省电", "低功耗", "绿色"))

        # 数量偏好
        count_match = re.search(r"(\d+)\s*(?:盏|个|只)\s*灯", text)
        lamp_count_pref = int(count_match.group(1)) if count_match else None

        result: Dict[str, Any] = {
            "raw_input": text,
            "color_temp": color_temp,
            "lux_override": lux_override,
            "lamp_type_preference": lamp_type_pref,
            "energy_saving": energy_saving,
            "lamp_count_preference": lamp_count_pref,
        }
        # 将解析结果写入 state 缓存，供后续工具使用
        state.tool_cache["user_requirement"] = result
        if lux_override:
            cached_req = state.tool_cache.get("room_requirement", {}) or {}
            cached_req["target_lux"] = lux_override
            state.tool_cache["room_requirement"] = cached_req
        state.record("tool_parse_user_requirement", {"user_input": text}, result)
        return result

    def tool_query_design_standard(
        self,
        state: RoomAgentState,
        query: str = "",
        top_k: int = 5,
        kg_store_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """检索照明设计规范条文（调用知识图谱混合检索）。"""
        resolved_query = (query or "").strip()
        if not resolved_query:
            resolved_query = f"{state.room_name}照明设计规范照度要求"

        kg_dir_str = kg_store_dir or os.getenv("KG_STORE_DIR", "kg_store")
        kg_path = Path(kg_dir_str)

        if not kg_path.exists():
            result: Dict[str, Any] = {
                "query": resolved_query,
                "status": "kg_not_found",
                "results": [],
                "message": f"知识图谱目录不存在: {kg_path}",
            }
            state.record("tool_query_design_standard", {"query": resolved_query}, result)
            return result

        try:
            from kg.hybrid_graph_query import hybrid_graph_query
            from kg.vector_store import requirement_index_exists

            # 构建 embed_fn（懒加载，缓存到 state）
            embed_fn: Optional[Callable] = state.tool_cache.get("_embed_fn")
            if embed_fn is None:
                embed_model_dir = os.getenv("EMBED_MODEL_DIR", "embed_model")
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(embed_model_dir)

                def _embed(texts: List[str]) -> np.ndarray:
                    vecs = _model.encode(texts, normalize_embeddings=True)
                    return np.array(vecs, dtype=np.float32)

                embed_fn = _embed
                state.tool_cache["_embed_fn"] = embed_fn

            if not requirement_index_exists(kg_path):
                result = {
                    "query": resolved_query,
                    "status": "index_not_built",
                    "results": [],
                    "message": "向量索引尚未构建，请先运行 kg pipeline",
                }
                state.record("tool_query_design_standard", {"query": resolved_query}, result)
                return result

            raw = hybrid_graph_query(resolved_query, kg_path, embed_fn, top_k=top_k)
            items = raw.get("results", []) or []
            simplified = [
                {
                    "id": item.get("id", ""),
                    "text": item.get("text", ""),
                    "source": item.get("source", ""),
                    "score": round(float(item.get("score", 0.0)), 4),
                    "expanded": bool(item.get("expanded", False)),
                }
                for item in items
            ]
            result = {
                "query": resolved_query,
                "status": "ok",
                "total": int(raw.get("total", len(simplified))),
                "results": simplified,
            }
        except Exception as exc:
            result = {
                "query": resolved_query,
                "status": "error",
                "results": [],
                "message": str(exc),
            }

        state.tool_cache["standard_query_result"] = result
        state.record("tool_query_design_standard", {"query": resolved_query, "top_k": top_k}, result)
        return result

    def tool_summarize_design(
        self,
        state: RoomAgentState,
    ) -> Dict[str, Any]:
        """汇总当前设计状态：灯具型号、数量、总功率、预估照度。"""
        lamps = state.placements.get("lamps", []) or []
        switches = state.placements.get("switches", []) or []
        lamp_count = len(lamps)
        switch_count = len(switches)

        lamp_model_plan = state.tool_cache.get("lamp_model_plan", {}) or {}
        selected_lamp = lamp_model_plan.get("selected_lamp", {}) or {}
        flux_plan = state.tool_cache.get("flux_plan", {}) or {}
        room_req = state.tool_cache.get("room_requirement", {}) or {}
        wiring_plan = state.tool_cache.get("wiring_plan", {}) or {}

        lamp_type = selected_lamp.get("lamp_type") or state.selected_lamp_type or "未知"
        lamp_model = selected_lamp.get("model", "未知")
        flux_per_lamp = float(selected_lamp.get("flux_lm", 0.0))
        power_per_lamp = selected_lamp.get("power_w") or 0.0
        total_power_w = float(power_per_lamp) * lamp_count if power_per_lamp else None

        area_m2 = max(0.01, float(state.area_m2))
        uf = float(flux_plan.get("uf", 0.6))
        mf = float(flux_plan.get("mf", 0.8))
        estimated_lux = None
        if flux_per_lamp > 0 and lamp_count > 0:
            estimated_lux = round((lamp_count * flux_per_lamp * uf * mf) / area_m2, 1)

        target_lux = int(room_req.get("target_lux", 300))
        lux_ok = None
        if estimated_lux is not None:
            lux_ok = bool(estimated_lux >= target_lux * 0.8)

        result = {
            "room_name": state.room_name,
            "area_m2": float(area_m2),
            "lamp_type": lamp_type,
            "lamp_model": lamp_model,
            "lamp_count": lamp_count,
            "switch_count": switch_count,
            "flux_per_lamp_lm": flux_per_lamp,
            "total_power_w": total_power_w,
            "estimated_lux": estimated_lux,
            "target_lux": target_lux,
            "lux_meets_target": lux_ok,
            "wiring_done": bool(wiring_plan.get("status") == "ok"),
            "wiring_route_count": int(wiring_plan.get("route_count", 0)),
        }
        state.record("tool_summarize_design", {}, result)
        return result

    def tool_check_standard_compliance(
        self,
        state: RoomAgentState,
        kg_store_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """对照规范检查当前设计是否合规（照度、功率密度）。"""
        summary = self.tool_summarize_design(state)
        issues: List[str] = []
        passed: List[str] = []

        # 照度合规检查
        estimated_lux = summary.get("estimated_lux")
        target_lux = summary.get("target_lux", 300)
        if estimated_lux is not None:
            if estimated_lux < target_lux * 0.8:
                issues.append(
                    f"预估照度{estimated_lux}lx低于目标照度{target_lux}lx的80%，不满足规范要求"
                )
            elif estimated_lux < target_lux:
                issues.append(f"预估照度{estimated_lux}lx略低于目标照度{target_lux}lx，建议增加灯具或选用更高光通量型号")
            else:
                passed.append(f"照度满足要求：预估{estimated_lux}lx ≥ 目标{target_lux}lx")

        # 功率密度检查（GB50034 参考限值：办公室≤9W/m²，普通房间≤7W/m²）
        total_power = summary.get("total_power_w")
        area_m2 = summary.get("area_m2", 1.0)
        if total_power and area_m2 > 0:
            lpd = total_power / area_m2
            lpd_limit = 9.0 if target_lux >= 300 else 7.0
            if lpd > lpd_limit:
                issues.append(f"照明功率密度{lpd:.1f}W/m²超过参考限值{lpd_limit}W/m²")
            else:
                passed.append(f"功率密度合规：{lpd:.1f}W/m² ≤ {lpd_limit}W/m²")

        # 尝试从 KG 检索相关规范条文
        standard_refs: List[str] = []
        try:
            kg_result = self.tool_query_design_standard(
                state,
                query=f"{state.room_name}照度标准功率密度",
                top_k=3,
                kg_store_dir=kg_store_dir,
            )
            for item in kg_result.get("results", [])[:3]:
                text = str(item.get("text", "")).strip()
                if text:
                    standard_refs.append(text[:120])
        except Exception:
            pass

        is_compliant = len(issues) == 0
        result = {
            "room_name": state.room_name,
            "is_compliant": is_compliant,
            "passed": passed,
            "issues": issues,
            "standard_refs": standard_refs,
            "design_summary": summary,
        }
        state.record("tool_check_standard_compliance", {}, result)
        return result

    def tool_diagnose_layout_issue(
        self,
        state: RoomAgentState,
    ) -> Dict[str, Any]:
        """分析当前布局存在的问题，给出具体修改建议。"""
        validation = self.tool_validate_layout(state)
        violations = validation.get("violations", []) or []
        stats = validation.get("stats", {}) or {}
        suggestions: List[str] = []
        diagnosis: List[Dict[str, str]] = []

        lamps = state.placements.get("lamps", []) or []
        switches = state.placements.get("switches", []) or []
        door_cells = [tuple(map(int, p)) for p in np.argwhere(state.matrix == 2).tolist()]
        valid_cells = [tuple(map(int, p)) for p in np.argwhere(state.matrix == 1).tolist()]

        for v in violations:
            code = v.get("code", "")
            if code == "NO_LAMP":
                diagnosis.append({"problem": "未放置灯具", "action": "调用 tool_place_components 放置灯具"})
            elif code == "NO_SWITCH":
                diagnosis.append({"problem": "未放置开关", "action": "调用 tool_place_components 并指定 switch_count≥1"})
            elif code == "LAMP_ON_INVALID_CELL":
                bad = [p for p in lamps if int(state.matrix[p[0], p[1]]) != 1]
                diagnosis.append({
                    "problem": f"灯具落在无效格: {bad[:3]}",
                    "action": "调用 tool_apply_layout_edit 将这些灯具移到值为1的格子",
                })
            elif code == "SWITCH_FAR_FROM_DOOR":
                if door_cells and switches:
                    nearest_door = min(door_cells, key=lambda d: abs(d[0]-switches[0][0])+abs(d[1]-switches[0][1]))
                    candidates = [
                        p for p in valid_cells
                        if abs(p[0]-nearest_door[0])+abs(p[1]-nearest_door[1]) <= 2
                    ]
                    if candidates:
                        suggestions.append(f"建议将开关移至 {candidates[0]} (靠近门格 {nearest_door})")
                diagnosis.append({
                    "problem": "开关距门过远",
                    "action": f"调用 tool_apply_layout_edit 将开关移到门附近，{suggestions[-1] if suggestions else ''}",
                })
            elif code == "UNEVEN_COVERAGE":
                cv = stats.get("coverage_cv", 0)
                diagnosis.append({
                    "problem": f"照明分布不均匀(cv={cv:.2f})",
                    "action": "调用 tool_place_components 重新布置，或调用 tool_apply_layout_edit 将灯具向房间中心均匀分散",
                })
            elif code == "LAMPS_TOO_CLOSE":
                min_d = stats.get("min_lamp_dist_cells", 0)
                diagnosis.append({
                    "problem": f"灯具间距过近(min={min_d:.1f}格)",
                    "action": "调用 tool_apply_layout_edit 拉大灯具间距，或减少灯具数量后重新布置",
                })

        if not diagnosis:
            diagnosis.append({"problem": "无明显问题", "action": "可直接调用 tool_generate_wiring 生成布线"})

        result = {
            "score": validation.get("score", 100),
            "is_valid": validation.get("is_valid", True),
            "diagnosis": diagnosis,
            "suggestions": suggestions,
            "violation_count": len(violations),
        }
        state.record("tool_diagnose_layout_issue", {}, result)
        return result

    def tool_generate_report(
        self,
        state: RoomAgentState,
    ) -> Dict[str, Any]:
        """生成照明设计说明书（markdown 格式）。"""
        summary = self.tool_summarize_design(state)
        validation = self.tool_validate_layout(state)
        tool_history = state.tool_history or []

        # 收集决策步骤
        steps: List[str] = []
        for record in tool_history:
            name = record.get("tool", "")
            if name in (
                "tool_lookup_room_requirement",
                "tool_estimate_component_count",
                "tool_calc_required_flux_per_lamp",
                "tool_retrieve_lamp_model",
                "tool_place_components",
                "tool_validate_layout",
                "tool_generate_wiring",
            ):
                steps.append(f"- 执行 `{name}`")

        lines = [
            f"# 照明设计报告 — {state.room_name}",
            "",
            "## 房间信息",
            f"- 房间名称：{state.room_name}",
            f"- 房间面积：{summary['area_m2']:.1f} m²",
            "",
            "## 灯具选型",
            f"- 灯具类型：{summary['lamp_type']}",
            f"- 灯具型号：{summary['lamp_model']}",
            f"- 单灯光通量：{summary['flux_per_lamp_lm']:.0f} lm",
            f"- 灯具数量：{summary['lamp_count']} 盏",
            f"- 开关数量：{summary['switch_count']} 个",
            "",
            "## 照度计算",
            f"- 目标照度：{summary['target_lux']} lx",
            f"- 预估照度：{summary['estimated_lux']} lx" if summary['estimated_lux'] else "- 预估照度：待计算",
            f"- 是否满足要求：{'是' if summary.get('lux_meets_target') else '否'}",
            f"- 总安装功率：{summary['total_power_w']:.0f} W" if summary['total_power_w'] else "- 总安装功率：未知",
            "",
            "## 布线",
            f"- 布线状态：{'完成' if summary['wiring_done'] else '未完成'}",
            f"- 布线路径数：{summary['wiring_route_count']}",
            "",
            "## 布局校验",
            f"- 评分：{validation.get('score', 0)} / 100",
            f"- 是否通过：{'是' if validation.get('is_valid') else '否'}",
        ]
        violations = validation.get("violations", []) or []
        if violations:
            lines.append("- 违规项：")
            for v in violations:
                lines.append(f"  - [{v.get('severity','').upper()}] {v.get('message','')}")

        if steps:
            lines += ["", "## 设计步骤", *steps]

        report_md = "\n".join(lines)
        result = {
            "room_name": state.room_name,
            "report_markdown": report_md,
            "summary": summary,
        }
        state.record("tool_generate_report", {}, {"room_name": state.room_name, "report_length": len(report_md)})
        return result

    def tool_read_matrix_state(
        self,
        state: RoomAgentState,
        max_rows: int = 64,
        max_cols: int = 64,
        compress: bool = True,
    ) -> Dict[str, Any]:
        board = state.to_ascii_board(max_rows=max_rows, max_cols=max_cols, compress=compress)
        result = {
            "summary": state.summary(),
            "ascii_board": board,
        }
        state.record(
            "tool_read_matrix_state",
            {"max_rows": max_rows, "max_cols": max_cols, "compress": compress},
            {"summary": result["summary"]},
        )
        return result

    def invoke_tool(
        self,
        tool_name: str,
        state: RoomAgentState,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = args or {}

        if tool_name == "tool_lookup_room_requirement":
            return self.tool_lookup_room_requirement(
                state=state,
                room_name=args.get("room_name"),
            )
        if tool_name == "tool_estimate_component_count":
            return self.tool_estimate_component_count(
                state=state,
                is_regular=args.get("is_regular"),
                min_spacing_m=float(args.get("min_spacing_m", 2.0)),
                max_spacing_m=float(args.get("max_spacing_m", 3.0)),
                max_lamps=int(args.get("max_lamps", 64)),
                switch_count=args.get("switch_count"),
            )
        if tool_name == "tool_calc_required_flux_per_lamp":
            return self.tool_calc_required_flux_per_lamp(
                state=state,
                target_lux=args.get("target_lux"),
                lamp_count=args.get("lamp_count"),
                uf=float(args.get("uf", 0.6)),
                mf=float(args.get("mf", 0.8)),
            )
        if tool_name == "tool_retrieve_lamp_model":
            return self.tool_retrieve_lamp_model(
                state=state,
                lamp_type=args.get("lamp_type"),
                required_flux_lm=args.get("required_flux_lm"),
                top_k=int(args.get("top_k", 3)),
            )
        if tool_name == "tool_place_components":
            return self.tool_place_components(
                state=state,
                lamp_count=args.get("lamp_count"),
                switch_count=args.get("switch_count"),
                is_regular=args.get("is_regular"),
            )
        if tool_name == "tool_validate_layout":
            return self.tool_validate_layout(
                state=state,
                min_lamp_dist_cells=int(args.get("min_lamp_dist_cells", 2)),
                max_switch_to_door_dist_cells=int(args.get("max_switch_to_door_dist_cells", 3)),
            )
        if tool_name == "tool_generate_wiring":
            return self.tool_generate_wiring(
                state=state,
                turn_penalty=args.get("turn_penalty"),
                bbox_pixel=args.get("bbox_pixel"),
                cell_size_px=int(args.get("cell_size_px", 40)),
                cad_params=args.get("cad_params"),
                image_width=args.get("image_width"),
                image_height=args.get("image_height"),
            )
        if tool_name == "tool_apply_layout_edit":
            return self.tool_apply_layout_edit(
                state=state,
                edits=args.get("edits", []),
            )
        if tool_name == "tool_read_matrix_state":
            return self.tool_read_matrix_state(
                state=state,
                max_rows=int(args.get("max_rows", 64)),
                max_cols=int(args.get("max_cols", 64)),
                compress=bool(args.get("compress", True)),
            )
        if tool_name == "tool_parse_user_requirement":
            return self.tool_parse_user_requirement(
                state=state,
                user_input=str(args.get("user_input", "")),
            )
        if tool_name == "tool_query_design_standard":
            return self.tool_query_design_standard(
                state=state,
                query=str(args.get("query", "")),
                top_k=int(args.get("top_k", 5)),
                kg_store_dir=args.get("kg_store_dir"),
            )
        if tool_name == "tool_summarize_design":
            return self.tool_summarize_design(state=state)
        if tool_name == "tool_check_standard_compliance":
            return self.tool_check_standard_compliance(
                state=state,
                kg_store_dir=args.get("kg_store_dir"),
            )
        if tool_name == "tool_diagnose_layout_issue":
            return self.tool_diagnose_layout_issue(state=state)
        if tool_name == "tool_generate_report":
            return self.tool_generate_report(state=state)
        raise ValueError(f"unknown tool: {tool_name}")


def _normalize_schema_for_anthropic(node: Any) -> Any:
    if isinstance(node, dict):
        normalized: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "type" and isinstance(value, list):
                non_null_types = [item for item in value if item != "null"]
                normalized[key] = non_null_types[0] if non_null_types else "string"
            else:
                normalized[key] = _normalize_schema_for_anthropic(value)
        return normalized
    if isinstance(node, list):
        return [_normalize_schema_for_anthropic(item) for item in node]
    return node


def build_anthropic_tools(tools: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    source_tools = tools or REACT_TOOLS
    anthropic_tools: List[Dict[str, Any]] = []
    for item in source_tools:
        if not isinstance(item, dict):
            continue
        function_block = item.get("function", {}) or {}
        name = str(function_block.get("name", "")).strip()
        if not name:
            continue
        anthropic_tools.append(
            {
                "name": name,
                "description": str(function_block.get("description", "")).strip(),
                "input_schema": _normalize_schema_for_anthropic(
                    deepcopy(function_block.get("parameters", {}) or {})
                ),
            }
        )
    return anthropic_tools


REACT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tool_lookup_room_requirement",
            "description": "查询房间照度要求和默认灯具类型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_name": {
                        "type": "string",
                        "description": "房间名称；不传时使用当前房间名称。",
                    }
                },
                "required": ["room_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_estimate_component_count",
            "description": "估算灯具数量、开关数量和推荐阵列。",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_regular": {"type": "boolean", "description": "是否规则房间。"},
                    "min_spacing_m": {"type": "number", "description": "最小期望间距(米)。"},
                    "max_spacing_m": {"type": "number", "description": "最大期望间距(米)。"},
                    "max_lamps": {"type": "integer", "description": "灯具数量上限。"},
                    "switch_count": {"type": "integer", "description": "显式指定开关数量。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_calc_required_flux_per_lamp",
            "description": "根据面积、目标照度和灯具数量计算单灯所需光通量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_lux": {"type": "integer", "description": "目标照度(lx)。"},
                    "lamp_count": {"type": "integer", "description": "灯具数量。"},
                    "uf": {"type": "number", "description": "利用系数。"},
                    "mf": {"type": "number", "description": "维护系数。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_retrieve_lamp_model",
            "description": "按灯具类型和所需光通量检索最匹配的灯具型号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lamp_type": {"type": "string", "description": "灯具类型。"},
                    "required_flux_lm": {"type": "number", "description": "目标光通量(lm)。"},
                    "top_k": {"type": "integer", "description": "返回候选数量。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_place_components",
            "description": "在离散网格中放置灯具和开关。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lamp_count": {"type": "integer", "description": "目标灯具数量。"},
                    "switch_count": {"type": "integer", "description": "目标开关数量。"},
                    "is_regular": {"type": "boolean", "description": "是否规则房间。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_validate_layout",
            "description": "校验当前布局是否合法、均匀，是否靠近门。",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_lamp_dist_cells": {"type": "integer", "description": "灯具最小间距(格)。"},
                    "max_switch_to_door_dist_cells": {
                        "type": "integer",
                        "description": "开关距门最大曼哈顿距离(格)。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_generate_wiring",
            "description": "基于开关和灯具位置生成布线路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "turn_penalty": {"type": "number", "description": "转弯惩罚。"},
                    "bbox_pixel": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "房间像素包围盒[min_x,min_y,max_x,max_y]。",
                    },
                    "cell_size_px": {"type": "integer", "description": "网格尺寸(像素)。"},
                    "cad_params": {"type": "object", "description": "CAD坐标换算参数。"},
                    "image_width": {"type": "integer", "description": "原图宽度。"},
                    "image_height": {"type": "integer", "description": "原图高度。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_apply_layout_edit",
            "description": "微调灯具或开关位置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "description": "批量编辑动作。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "component_type": {
                                    "type": "string",
                                    "enum": ["lamps", "switches"],
                                    "description": "元件类型。",
                                },
                                "source": {
                                    "type": ["array", "null"],
                                    "items": {"type": "integer"},
                                    "description": "原位置[row,col]，可为空。",
                                },
                                "target": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "目标位置[row,col]。",
                                },
                            },
                            "required": ["component_type", "target"],
                        },
                    }
                },
                "required": ["edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_read_matrix_state",
            "description": "读取当前房间的ASCII棋盘和布局摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_rows": {"type": "integer", "description": "ASCII最大行数。"},
                    "max_cols": {"type": "integer", "description": "ASCII最大列数。"},
                    "compress": {"type": "boolean", "description": "是否使用RLE压缩。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_parse_user_requirement",
            "description": "解析用户自然语言需求，提取色温、照度、灯具类型等结构化约束。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "用户原始需求文本。"},
                },
                "required": ["user_input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_query_design_standard",
            "description": "从知识图谱检索照明设计规范条文（照度标准、功率密度限值等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，如'办公室照度标准'。"},
                    "top_k": {"type": "integer", "description": "返回条文数量，默认5。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_summarize_design",
            "description": "汇总当前设计状态：灯具型号、数量、总功率、预估照度。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_check_standard_compliance",
            "description": "对照规范检查当前设计是否合规（照度、功率密度），并引用相关规范条文。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_diagnose_layout_issue",
            "description": "分析当前布局存在的问题，给出具体可执行的修改建议。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_generate_report",
            "description": "生成完整的照明设计说明书（markdown格式），包含选型、照度计算、布线和合规说明。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


REACT_SYSTEM_PROMPT_TEMPLATE = """
你是建筑照明自动化智能体。目标是在离散网格上完成:
需求匹配 -> 数量估计 -> 光通量计算 -> 型号检索 -> 布点 -> 校验修正 -> 布线。

你不会使用模型内置函数调用。你只能阅读下面的工具数组定义，然后自行输出一个JSON动作对象。
每次工具执行后，系统会返回自然语言形式的 `tool_result` 给你。你必须根据 `tool_result` 继续推理，不能伪造工具结果。

可用工具数组如下:
{tools_block}

可选动作名:
{tool_names}, finish

网格语义:
- matrix中: 0=障碍, 1=可布置, 2=门位
- ascii_board中: #=障碍, .=可布置, D=门, L=灯, S=开关

输出要求:
- 每轮只能输出一个JSON对象。
- 不要输出Markdown，不要输出解释，不要输出多段文本。
- action 必须严格等于一个工具名或 finish。
- args 必须匹配对应工具的 parameters 定义；不需要的可选参数可以省略。

固定输出格式:
{{
  "thought": "一句简短决策理由",
  "action": "某个工具名或finish",
  "args": {{}},
  "reason": "仅在action=finish时必填",
  "strategy": "仅在action=finish时必填"
}}

执行原则:
- 完整设计流程分四个阶段，按顺序推进:
  【需求解析】tool_parse_user_requirement(有用户输入时) -> tool_query_design_standard -> tool_lookup_room_requirement
  【设计】tool_estimate_component_count -> tool_calc_required_flux_per_lamp -> tool_retrieve_lamp_model -> tool_place_components
  【检查与修正】tool_validate_layout -> tool_diagnose_layout_issue(若有违规) -> tool_apply_layout_edit 或重新 tool_place_components
  【收尾】tool_generate_wiring -> tool_check_standard_compliance -> tool_generate_report -> finish
- 若校验失败或有明显 warning，调用 tool_diagnose_layout_issue 获取具体修改建议，再执行修改。
- 布线前必须至少有1个开关和1个灯具。
- 仅当布局、布线、报告都完成时调用 finish。
- 任何 target 都必须落在值为1的网格；不能放到0或2上。
"""


MINIMAX_TOOL_AGENT_SYSTEM_PROMPT = """
你是建筑照明自动化智能体。你的任务是在单个房间的离散网格上，完成完整的照明设计闭环：

【阶段一：需求解析】
1. 若有用户输入，调用 tool_parse_user_requirement 解析需求约束。
2. 调用 tool_query_design_standard 检索相关照明规范。
3. 调用 tool_lookup_room_requirement 查询房间照度要求与默认灯具类型。

【阶段二：设计】
4. 估算灯具数量、开关数量和推荐阵列。
5. 计算单灯所需光通量。
6. 检索合适的灯具型号。
7. 在网格中放置灯具和开关。

【阶段三：检查与修正】
8. 调用 tool_validate_layout 校验布局。
9. 若有违规，调用 tool_diagnose_layout_issue 获取具体修改建议，再执行修改。

【阶段四：收尾】
10. 布置完成后生成布线。
11. 调用 tool_check_standard_compliance 进行合规检查。
12. 调用 tool_generate_report 生成设计报告。

你可以直接调用系统提供的工具，不要伪造工具结果。
工具返回会以自然语言 Observation 的形式回传给你；其中读取矩阵状态的工具还会返回 ASCII 棋盘。

网格语义：
- matrix 中: 0=障碍, 1=可布置, 2=门位
- ASCII 棋盘中: #=障碍, .=可布置, D=门, L=灯, S=开关

执行原则：
- 如果校验失败或存在 warning，调用 tool_diagnose_layout_issue 后再修正，不要盲目重试。
- 在调用布线工具前，必须保证至少已有 1 个灯具和 1 个开关。
- 当布局、布线、报告都完成后，直接给出最终中文总结，不要再调用工具。

最终总结至少包含：
- 房间名称、灯具类型、灯具数量、开关数量
- 预估照度与目标照度对比
- 合规检查结论
- 布线是否完成
"""


DEFAULT_LAMP_CATALOG: List[Dict[str, Any]] = [
    {
        "灯具类型": "感应式吸顶灯",
        "型号": "LPXDD 002",
        "光通量": "1000lm",
        "功率": "15W",
        "厂家": "Alibaba 供应商",
        "购买链接": "https://www.alibaba.com/product-detail/Modern-Intelligent-LED-Induction-Ceiling-Light_1601488477089.html",
    },
    {
        "灯具类型": "防爆灯",
        "型号": "BC9102S-L30",
        "光通量": "4200lm",
        "功率": "30W",
        "厂家": "通明电器 TORMIN",
        "购买链接": "https://i-item.jd.com/100021096200.html",
    },
    {
        "灯具类型": "双管格栅灯",
        "型号": "ML-XTD014E",
        "光通量": "3500lm",
        "功率": "36W",
        "厂家": "Moonlight",
        "购买链接": "https://www.alibaba.com/product-detail/Industrial-Grille-Light-36W-4FT-T8_1600967544017.html",
    },
    {
        "灯具类型": "双管荧光灯",
        "型号": "BAY51-S28XJWF1",
        "光通量": "5600lm",
        "功率": "56W",
        "厂家": "合隆 Helon",
        "购买链接": "https://test-www.mymro.cn:443/u-8W2652.html",
    },
    {
        "灯具类型": "筒灯",
        "型号": "tp2351q",
        "光通量": "540lm",
        "功率": "6W",
        "厂家": "tp",
        "购买链接": "https://www.alibaba.com/product-detail/Modern-Aluminum-Recessed-Downlight-Led-Spotlight_1601702044947.html",
    },
]


DEFAULT_ROOM_LUX: Dict[str, int] = {
    "办公室": 300,
    "办公室1": 300,
    "楼梯间": 100,
    "楼梯间1": 100,
    "配电室": 200,
    "煤样存放室": 100,
    "元素分析室": 500,
    "高温室": 300,
    "热量室准备间": 300,
    "工业分析室": 500,
    "天平室": 500,
    "接样室": 300,
    "男卫生间": 100,
    "备用间": 100,
    "盟洗室": 100,
    "存样室": 100,
    "女卫生间": 100,
    "除尘室": 300,
}
