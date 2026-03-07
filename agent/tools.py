from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
                    "has_cad": bool(cad_params),
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
        can_export_cad = bool(can_export_pixel and cad_params and image_width and image_height)

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
                        cad_params=cad_params or {},
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
                        cad_params=cad_params or {},
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
                "has_cad": bool(cad_params),
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
                "required": [],
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
- 从空布局开始时，优先顺序为:
  tool_lookup_room_requirement -> tool_estimate_component_count -> tool_calc_required_flux_per_lamp -> tool_retrieve_lamp_model -> tool_place_components -> tool_validate_layout。
- 若校验失败或有明显 warning，优先调用 tool_apply_layout_edit 或重新调用 tool_place_components。
- 布线前必须至少有1个开关和1个灯具。
- 布局可用后再调用 tool_generate_wiring。
- 仅当布局和布线都完成时调用 finish。
- 任何 target 都必须落在值为1的网格；不能放到0或2上。
"""


MINIMAX_TOOL_AGENT_SYSTEM_PROMPT = """
你是建筑照明自动化智能体。你的任务是在单个房间的离散网格上，完成完整的照明设计闭环：
1. 查询房间照度要求与默认灯具类型。
2. 估算灯具数量、开关数量和推荐阵列。
3. 计算单灯所需光通量。
4. 检索合适的灯具型号。
5. 在网格中放置灯具和开关。
6. 校验布局；如果有问题，继续调整位置或重新布置。
7. 布置完成后生成布线。

你可以直接调用系统提供的工具，不要伪造工具结果。
工具返回会以自然语言 Observation 的形式回传给你；其中读取矩阵状态的工具还会返回 ASCII 棋盘。

网格语义：
- matrix 中: 0=障碍, 1=可布置, 2=门位
- ASCII 棋盘中: #=障碍, .=可布置, D=门, L=灯, S=开关

执行原则：
- 从空布局开始时，优先顺序为：
  查询需求 -> 估算数量 -> 计算光通量 -> 检索型号 -> 放置元件 -> 校验布局。
- 如果校验失败或存在 warning，优先重新放置或调用位置调整工具修正。
- 在调用布线工具前，必须保证至少已有 1 个灯具和 1 个开关。
- 当你确认布局和布线都完成后，直接给出最终中文总结，不要再调用工具。

最终总结至少包含：
- 房间名称
- 灯具类型
- 灯具数量
- 开关数量
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
