"""
CAD分析服务主逻辑模块
处理外部请求，调用核心分析功能，返回房间CAD坐标
"""
import copy
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.state import RoomAgentState
from .find_all import process_images_batch, process_single_image, process_layout_from_intermediate
from .coordinate_converter import DEFAULT_CAD_PARAMS, pixel_to_cad
from .lighting_layout import (
    LAMP_CATALOG_DEFAULT,
    _complete_lamp_cells,
    _get_target_lux_for_room,
    _grid_cell_to_pixel,
    _parse_luminous_flux_to_lm,
    _select_lamp_cells_regular_grid,
    _select_lamp_cells_rule_based,
    _select_switch_near_door,
    process_room_lighting_layout,
)
from .logger import get_logger
from .wiring_layout import process_room_wiring_layout

# 获取logger实例
logger = get_logger("cad_service")
SESSION_ROOT = Path(__file__).resolve().parents[1] / "cad_sessions"


class CADParams(BaseModel):
    """CAD参数数据模型"""
    Xmin: float = Field(..., description="CAD窗口左边界")
    Ymin: float = Field(..., description="CAD窗口下边界")  
    Xmax: float = Field(..., description="CAD窗口右边界")
    Ymax: float = Field(..., description="CAD窗口上边界")

    @field_validator('Xmin', 'Ymin', 'Xmax', 'Ymax')
    def validate_bounds(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('坐标值必须是数字')
        return float(v)


class CADRequest(BaseModel):
    """CAD分析请求数据模型"""
    image_directory: str = Field(..., description="PNG图片目录路径")
    cad_params: Optional[CADParams] = Field(None, description="CAD参数，为空时使用默认参数")

    @field_validator('image_directory')
    def validate_directory(cls, v):
        if not v or not v.strip():
            raise ValueError('图片目录路径不能为空')
        return v.strip()


class RoomCADCoordinate(BaseModel):
    """房间CAD坐标数据模型"""
    room_name: str = Field(..., description="房间名称")
    cad_coordinates: List[List[List[float]]] = Field(..., description="CAD坐标点列表")


class LampsPlan(BaseModel):
    """房间统一灯具配置（同类型灯具聚合）"""
    lamp_type: str = Field(..., description="灯具类型")
    count: int = Field(..., description="灯具数量")
    cad_positions: List[List[float]] = Field(default_factory=list, description="所有灯具CAD坐标[[x,y], ...]")


class SwitchPlacement(BaseModel):
    """开关放置点数据模型"""
    switch_type: str = Field("开关", description="开关类型")
    cad_position: List[float] = Field(default_factory=list, description="CAD坐标[x, y]")


class RoomLightingPlan(BaseModel):
    """房间灯具布置结果"""
    room_name: str = Field(..., description="房间名称")
    lamp_count: int = Field(..., description="灯具数量")
    room_area_m2: float = Field(0.0, description="房间面积（平方米）")
    lamps: LampsPlan = Field(default_factory=lambda: LampsPlan(lamp_type="筒灯", count=0), description="灯具布置")
    switch: Optional[SwitchPlacement] = Field(default=None, description="开关布置点")
    switch_count: int = Field(0, description="开关数量")


class WiringSegment(BaseModel):
    """布线线段（用于CAD插入）"""
    start_cad: List[float] = Field(default_factory=list, description="起点CAD坐标[x, y, z]")
    end_cad: List[float] = Field(default_factory=list, description="终点CAD坐标[x, y, z]")
    layer_name: str = Field("wiringlayer", description="线段图层名")
    line_width: float = Field(20.0, description="线宽")
    color: str = Field("yellow", description="颜色")


class RoomWiringPlan(BaseModel):
    """房间布线结果"""
    room_name: str = Field(..., description="房间名称")
    status: str = Field("ok", description="布线状态: ok/partial/skipped")
    segment_count: int = Field(0, description="线段数量")
    segments: List[WiringSegment] = Field(default_factory=list, description="布线线段列表")


class CADResponse(BaseModel):
    """CAD分析响应数据模型"""
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="处理结果消息")
    session_id: Optional[str] = Field(default=None, description="服务端会话ID，用于后续执行步骤7和步骤8")
    total_images: int = Field(..., description="处理的图片总数")
    processed_images: int = Field(..., description="成功处理的图片数")
    results: Dict[str, List[RoomCADCoordinate]] = Field(..., description="每个图片的房间CAD坐标结果")
    lighting_results: Dict[str, List[RoomLightingPlan]] = Field(
        default_factory=dict,
        description="每个图片的房间灯具布置结果",
    )
    wiring_results: Dict[str, List[RoomWiringPlan]] = Field(
        default_factory=dict,
        description="每个图片的房间布线结果",
    )
    errors: Dict[str, str] = Field(default_factory=dict, description="处理错误信息")


class CADLayoutResponse(BaseModel):
    """步骤7和步骤8处理响应数据模型"""
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="处理结果消息")
    session_id: str = Field(..., description="服务端会话ID")
    total_images: int = Field(..., description="处理的图片总数")
    processed_images: int = Field(..., description="成功处理的图片数")
    lighting_results: Dict[str, List[RoomLightingPlan]] = Field(
        default_factory=dict,
        description="每个图片的房间灯具布置结果",
    )
    wiring_results: Dict[str, List[RoomWiringPlan]] = Field(
        default_factory=dict,
        description="每个图片的房间布线结果",
    )
    errors: Dict[str, str] = Field(default_factory=dict, description="处理错误信息")


class ChatTraceEntry(BaseModel):
    """chat turn 结构化执行轨迹"""
    step: int = Field(..., description="轨迹步骤编号")
    thought: str = Field("", description="思考文本")
    action: str = Field(..., description="动作名称")
    action_input: Dict[str, Any] = Field(default_factory=dict, description="动作输入")
    observation: str = Field("", description="观察结果")


class ChatRoomState(BaseModel):
    """chat turn 房间持久化状态"""
    model_config = ConfigDict(extra="allow")

    session_id: str = Field(..., description="服务端会话ID")
    conversation_id: str = Field(..., description="对话ID")
    room_name: str = Field(..., description="房间名称")
    image_name: str = Field(..., description="所属图片名称")
    image_path: str = Field(..., description="所属图片路径")
    selected_lamp_type: str = Field(..., description="当前选中的灯具类型")
    lamp_count: int = Field(..., description="当前灯具数量")
    target_lux: float = Field(..., description="目标照度")
    required_flux_per_lamp_lm: Optional[float] = Field(default=None, description="单灯所需光通量")
    lamp_model: Optional[str] = Field(default=None, description="灯具型号")
    placement_mode: str = Field(..., description="布置模式: rule/llm")
    lamps: List[List[int]] = Field(default_factory=list, description="灯具网格坐标列表")
    switch: Optional[Dict[str, Any]] = Field(default=None, description="开关信息")
    lighting_result: RoomLightingPlan = Field(..., description="对外灯具结果")
    lighting_internal: Dict[str, Any] = Field(default_factory=dict, description="内部灯具状态")
    wiring: RoomWiringPlan = Field(..., description="内部布线结果")
    validation: Dict[str, Any] = Field(default_factory=dict, description="校验结果")
    tool_cache: Dict[str, Any] = Field(default_factory=dict, description="工具缓存")
    execution_history: List[Dict[str, Any]] = Field(default_factory=list, description="执行历史")
    updated_at: str = Field(..., description="更新时间")


class ChatTurnResponse(BaseModel):
    """chat turn 接口响应"""
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="处理结果消息")
    session_id: str = Field(..., description="服务端会话ID")
    conversation_id: str = Field(..., description="对话ID")
    intent_type: str = Field(..., description="本轮意图类型")
    room_name: str = Field(..., description="目标房间名称")
    image_name: str = Field(..., description="所属图片名称")
    start_from: str = Field(..., description="后端推断的执行起点")
    lighting_result: RoomLightingPlan = Field(..., description="单房间灯具结果")
    wiring_result: RoomWiringPlan = Field(..., description="单房间布线结果")
    room_state: ChatRoomState = Field(..., description="当前房间完整状态")
    trace: List[ChatTraceEntry] = Field(default_factory=list, description="结构化执行轨迹")


class CADAnalysisService:
    """CAD分析服务类"""
    
    def __init__(self):
        self.default_cad_params = DEFAULT_CAD_PARAMS
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    
    def validate_request(self, request: CADRequest) -> tuple[bool, str]:
        """
        验证请求参数
        :param request: CAD分析请求
        :return: (是否有效, 错误消息)
        """
        # 验证目录是否存在
        if not os.path.exists(request.image_directory):
            return False, f"指定目录不存在: {request.image_directory}"
        
        if not os.path.isdir(request.image_directory):
            return False, f"指定路径不是目录: {request.image_directory}"
        
        # 检查目录是否包含PNG文件
        png_files = [f for f in os.listdir(request.image_directory) 
                    if f.lower().endswith('.png')]
        
        if not png_files:
            return False, f"目录中没有找到PNG文件: {request.image_directory}"
        
        return True, ""
    
    def validate_uploaded_files(self, files: List[bytes], filenames: List[str]) -> tuple[bool, str]:
        """
        验证上传的文件
        :param files: 文件内容列表
        :param filenames: 文件名列表
        :return: (是否有效, 错误消息)
        """
        if not files:
            return False, "没有上传任何文件"
        
        # 验证文件格式
        for filename in filenames:
            if not filename.lower().endswith('.png'):
                return False, f"不支持的文件格式: {filename}，只支持PNG格式"
        
        # 验证文件大小（限制为50MB每个文件）
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        for i, file_content in enumerate(files):
            if len(file_content) > MAX_FILE_SIZE:
                return False, f"文件 {filenames[i]} 大小超过限制 (50MB)"
        
        return True, ""
    
    def _create_session_workspace(self) -> tuple[str, Path]:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        session_dir = SESSION_ROOT / session_id
        uploads_dir = session_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=False)
        return session_id, session_dir

    def _session_manifest_path(self, session_id: str) -> Path:
        return SESSION_ROOT / session_id / "manifest.json"

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _write_session_manifest(self, session_id: str, payload: Dict[str, Any]) -> None:
        manifest_path = self._session_manifest_path(session_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2)

    def _read_session_manifest(self, session_id: str) -> Dict[str, Any]:
        manifest_path = self._session_manifest_path(session_id)
        if not manifest_path.exists():
            raise FileNotFoundError(f"会话不存在: {session_id}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"会话清单格式错误: {manifest_path}")
        return data

    def _chat_root_dir(self, session_id: str) -> Path:
        return SESSION_ROOT / session_id / "chat"

    def _conversation_dir(self, session_id: str, conversation_id: str) -> Path:
        return self._chat_root_dir(session_id) / conversation_id

    def _conversation_path(self, session_id: str, conversation_id: str) -> Path:
        return self._conversation_dir(session_id, conversation_id) / "conversation.json"

    def _sanitize_room_filename(self, room_name: str) -> str:
        normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", (room_name or "").strip())
        normalized = normalized.strip("._") or "room"
        return f"{normalized[:48]}_{uuid.uuid5(uuid.NAMESPACE_URL, room_name).hex[:10]}.json"

    def _room_state_path(self, session_id: str, conversation_id: str, room_name: str) -> Path:
        return self._conversation_dir(session_id, conversation_id) / "rooms" / self._sanitize_room_filename(room_name)

    def _write_json_file(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2)

    def _read_json_file(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"JSON格式错误: {path}")
        return data

    def _list_available_rooms(self, manifest: Dict[str, Any]) -> List[Dict[str, str]]:
        images = manifest.get("images") or {}
        rooms: List[Dict[str, str]] = []
        for image_name, image_payload in images.items():
            if not isinstance(image_payload, dict):
                continue
            intermediate = image_payload.get("intermediate") or {}
            if not isinstance(intermediate, dict):
                continue
            room_rectangles = intermediate.get("room_rectangles") or {}
            cad_rooms = intermediate.get("cad_rooms") or {}
            if not isinstance(room_rectangles, dict):
                room_rectangles = {}
            if not isinstance(cad_rooms, dict):
                cad_rooms = {}
            room_names = sorted({str(name) for name in room_rectangles.keys()} | {str(name) for name in cad_rooms.keys()})
            for room_name in room_names:
                rooms.append({"room_name": room_name, "image_name": str(image_name)})
        return rooms

    def _list_conversation_ids(self, session_id: str) -> List[str]:
        chat_root = self._chat_root_dir(session_id)
        if not chat_root.exists():
            return []
        ids = []
        for item in chat_root.iterdir():
            if item.is_dir() and (item / "conversation.json").exists():
                ids.append(item.name)
        return sorted(ids)

    def _load_conversation(
        self,
        session_id: str,
        conversation_id: Optional[str],
        *,
        create: bool,
    ) -> tuple[str, Dict[str, Any]]:
        resolved_id = (conversation_id or "").strip()
        if not resolved_id:
            if not create:
                raise ValueError("conversation_id is required")
            resolved_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]

        conversation_path = self._conversation_path(session_id, resolved_id)
        if conversation_path.exists():
            conversation = self._read_json_file(conversation_path)
        else:
            if not create:
                raise FileNotFoundError(f"会话对话不存在: {resolved_id}")
            now = datetime.now().isoformat()
            conversation = {
                "session_id": session_id,
                "conversation_id": resolved_id,
                "current_room": None,
                "turn_count": 0,
                "history": [],
                "global_preferences": {},
                "created_at": now,
                "updated_at": now,
            }
            self._write_json_file(conversation_path, conversation)
        return resolved_id, conversation

    def _load_room_state(
        self,
        session_id: str,
        conversation_id: str,
        room_name: str,
    ) -> Optional[Dict[str, Any]]:
        room_path = self._room_state_path(session_id, conversation_id, room_name)
        if not room_path.exists():
            return None
        return self._read_json_file(room_path)

    def _load_all_room_states(self, session_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        rooms_dir = self._conversation_dir(session_id, conversation_id) / "rooms"
        if not rooms_dir.exists():
            return []
        states: List[Dict[str, Any]] = []
        for path in sorted(rooms_dir.glob("*.json")):
            try:
                states.append(self._read_json_file(path))
            except Exception as exc:
                logger.warning(f"读取房间状态失败 {path}: {exc}")
        return states

    def _resolve_room_context(
        self,
        manifest: Dict[str, Any],
        room_name: str,
    ) -> Dict[str, Any]:
        images = manifest.get("images") or {}
        cad_params_dict = manifest.get("cad_params") or self.default_cad_params
        matches: List[Dict[str, Any]] = []
        for image_name, image_payload in images.items():
            if not isinstance(image_payload, dict):
                continue
            intermediate = image_payload.get("intermediate") or {}
            if not isinstance(intermediate, dict):
                continue
            room_rectangles = intermediate.get("room_rectangles") or {}
            if not isinstance(room_rectangles, dict):
                continue
            if room_name not in room_rectangles:
                continue
            matches.append(
                {
                    "image_name": str(image_name),
                    "image_path": str(image_payload.get("image_path") or ""),
                    "room_shapes": room_rectangles.get(room_name),
                    "door_assignments": intermediate.get("door_assignments") or [],
                    "cad_params": cad_params_dict,
                }
            )

        if not matches:
            raise ValueError(f"房间不存在于当前session: {room_name}")
        if len(matches) > 1:
            image_names = [item["image_name"] for item in matches]
            raise ValueError(f"房间名不唯一，请先区分所属图像: {room_name} -> {image_names}")
        if not matches[0]["image_path"]:
            raise ValueError(f"房间缺少图像路径: {room_name}")
        return matches[0]

    def _trace_entry(
        self,
        step: int,
        thought: str,
        action: str,
        action_input: Optional[Dict[str, Any]],
        observation: str,
    ) -> Dict[str, Any]:
        return {
            "step": int(step),
            "thought": thought,
            "action": action,
            "action_input": self._json_safe(action_input or {}),
            "observation": observation,
        }

    def _emit_chat_event(
        self,
        event_sink: Optional[Callable[[Dict[str, Any]], None]],
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if event_sink is None:
            return
        event = {
            "type": str(event_type),
            "timestamp": datetime.now().isoformat(),
        }
        if isinstance(payload, dict):
            event.update(self._json_safe(payload))
        try:
            event_sink(event)
        except Exception as exc:
            logger.debug(f"emit chat event failed: {exc}")

    def _resolve_catalog_lamp(self, lamp_type: str, lamp_model: Optional[str]) -> Dict[str, Any]:
        model_name = (lamp_model or "").strip()
        lamp_name = (lamp_type or "").strip()

        selected = None
        if model_name:
            for item in LAMP_CATALOG_DEFAULT:
                if str(item.get("型号", "")).strip() == model_name:
                    selected = dict(item)
                    break
        if selected is None and lamp_name:
            for item in LAMP_CATALOG_DEFAULT:
                if str(item.get("灯具类型", "")).strip() == lamp_name:
                    selected = dict(item)
                    break

        phi_lm = None
        if selected is not None:
            phi_lm = _parse_luminous_flux_to_lm(str(selected.get("光通量", "")), default_lm=1000.0)

        resolved_lamp_type = lamp_name
        if not resolved_lamp_type and selected is not None:
            resolved_lamp_type = str(selected.get("灯具类型", "")).strip()
        resolved_lamp_model = model_name
        if not resolved_lamp_model and selected is not None:
            resolved_lamp_model = str(selected.get("型号", "")).strip() or None

        return {
            "lamp_type": resolved_lamp_type,
            "lamp_model": resolved_lamp_model,
            "phi_lm": float(phi_lm) if phi_lm is not None else None,
            "catalog_item": selected,
        }

    def _calc_required_flux_per_lamp(
        self,
        area_m2: float,
        target_lux: float,
        lamp_count: int,
        *,
        uf: float = 0.6,
        mf: float = 0.8,
    ) -> Optional[float]:
        if lamp_count <= 0:
            return None
        return float((float(target_lux) * float(area_m2)) / (max(0.05, float(uf)) * max(0.05, float(mf)) * float(lamp_count)))

    def _normalize_grid_points(
        self,
        points: Any,
        matrix: np.ndarray,
        *,
        allow_door_cells: bool,
    ) -> List[Tuple[int, int]]:
        if points is None:
            return []
        if not isinstance(points, list):
            raise ValueError("网格坐标必须是数组")

        rows, cols = matrix.shape
        normalized: List[Tuple[int, int]] = []
        seen = set()
        for item in points:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], (int, float))
                or not isinstance(item[1], (int, float))
            ):
                raise ValueError(f"无效网格坐标: {item}")
            row = int(item[0])
            col = int(item[1])
            if row < 0 or col < 0 or row >= rows or col >= cols:
                raise ValueError(f"网格坐标越界: {[row, col]}")

            cell_value = int(matrix[row, col])
            if allow_door_cells:
                if cell_value == 0:
                    raise ValueError(f"坐标不可用: {[row, col]}")
            elif cell_value != 1:
                raise ValueError(f"灯具坐标必须位于可布置网格: {[row, col]}")

            key = (row, col)
            if key in seen:
                raise ValueError(f"网格坐标重复: {[row, col]}")
            seen.add(key)
            normalized.append(key)
        return normalized

    def _select_lamp_cells_for_chat(
        self,
        base_internal: Dict[str, Any],
        *,
        lamp_count: int,
        placement_mode: str,
        explicit_lamps: Any,
        existing_state: Optional[Dict[str, Any]],
        preserve_existing: bool,
    ) -> tuple[List[Tuple[int, int]], str]:
        matrix = np.array(base_internal.get("matrix") or [], dtype=np.int32)
        if matrix.size == 0 or matrix.ndim != 2:
            return [], "empty"

        if explicit_lamps is not None:
            return self._normalize_grid_points(explicit_lamps, matrix, allow_door_cells=False), "explicit"

        if preserve_existing and existing_state:
            existing_positions = existing_state.get("lamps")
            if existing_positions is None:
                lighting_internal = existing_state.get("lighting_internal") or {}
                existing_positions = ((lighting_internal.get("lamps", {}) or {}).get("grid_positions")) or []
            try:
                normalized_existing = self._normalize_grid_points(
                    existing_positions,
                    matrix,
                    allow_door_cells=False,
                )
                if len(normalized_existing) == int(lamp_count):
                    return normalized_existing, "existing"
            except Exception:
                pass

        if placement_mode == "rule":
            if bool(base_internal.get("is_regular")):
                return _select_lamp_cells_regular_grid(matrix, lamp_count), "rule_regular"
            return _select_lamp_cells_rule_based(matrix, lamp_count), "rule_irregular"

        base_positions = ((base_internal.get("lamps", {}) or {}).get("grid_positions")) or []
        try:
            picked = self._normalize_grid_points(base_positions, matrix, allow_door_cells=False)
        except Exception:
            picked = []
        return _complete_lamp_cells(picked, matrix, lamp_count), "llm_base"

    def _select_switch_for_chat(
        self,
        base_internal: Dict[str, Any],
        *,
        explicit_switch: Any,
        existing_state: Optional[Dict[str, Any]],
        preserve_existing: bool,
    ) -> tuple[Optional[Tuple[int, int]], str]:
        matrix = np.array(base_internal.get("matrix") or [], dtype=np.int32)
        if matrix.size == 0 or matrix.ndim != 2:
            return None, "empty"

        if explicit_switch is not None:
            normalized = self._normalize_grid_points([explicit_switch], matrix, allow_door_cells=True)
            return normalized[0], "explicit"

        if preserve_existing and existing_state:
            existing_switch = existing_state.get("switch")
            if not isinstance(existing_switch, dict):
                lighting_internal = existing_state.get("lighting_internal") or {}
                existing_switch = lighting_internal.get("switch") or {}
            if isinstance(existing_switch, dict) and existing_switch.get("grid_position") is not None:
                try:
                    normalized = self._normalize_grid_points(
                        [existing_switch.get("grid_position")],
                        matrix,
                        allow_door_cells=True,
                    )
                    return normalized[0], "existing"
                except Exception:
                    pass

        switch_info = base_internal.get("switch") or {}
        if isinstance(switch_info, dict) and switch_info.get("grid_position") is not None:
            try:
                normalized = self._normalize_grid_points(
                    [switch_info.get("grid_position")],
                    matrix,
                    allow_door_cells=True,
                )
                return normalized[0], "base"
            except Exception:
                pass

        fallback = _select_switch_near_door(
            grid=matrix,
            door_edge_cells=base_internal.get("door_edge_cells") or [],
            door_side=base_internal.get("door_side"),
        )
        if fallback is None:
            return None, "none"
        return (int(fallback[0]), int(fallback[1])), "fallback"

    def _build_room_lighting_from_cells(
        self,
        *,
        room_name: str,
        base_internal: Dict[str, Any],
        image_width: int,
        image_height: int,
        cad_params: Dict[str, float],
        lamp_type: str,
        lamp_model: Optional[str],
        target_lux: float,
        required_flux_per_lamp_lm: Optional[float],
        placement_mode: str,
        lamp_cells: List[Tuple[int, int]],
        switch_cell: Optional[Tuple[int, int]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        bbox = base_internal.get("bbox_pixel") or []
        if len(bbox) != 4:
            raise ValueError(f"房间缺少bbox信息: {room_name}")
        min_x, min_y, max_x, max_y = [int(v) for v in bbox]
        room_w = max_x - min_x + 1
        room_h = max_y - min_y + 1
        cell_size_px = int(base_internal.get("cell_size_px", 40))

        lamp_grid_positions: List[List[int]] = []
        lamp_pixel_positions: List[List[float]] = []
        lamp_cad_positions: List[List[float]] = []
        for row, col in lamp_cells:
            px, py = _grid_cell_to_pixel(
                row=int(row),
                col=int(col),
                min_x=min_x,
                min_y=min_y,
                room_w=room_w,
                room_h=room_h,
                cell_size_px=cell_size_px,
            )
            x_cad, y_cad = pixel_to_cad(
                px=px,
                py=py,
                Xmin=cad_params["Xmin"],
                Ymin=cad_params["Ymin"],
                Xmax=cad_params["Xmax"],
                Ymax=cad_params["Ymax"],
                width=image_width,
                height=image_height,
            )
            lamp_grid_positions.append([int(row), int(col)])
            lamp_pixel_positions.append([float(px), float(py)])
            lamp_cad_positions.append([float(x_cad), float(y_cad)])

        switch_info = None
        if switch_cell is not None:
            srow, scol = int(switch_cell[0]), int(switch_cell[1])
            spx, spy = _grid_cell_to_pixel(
                row=srow,
                col=scol,
                min_x=min_x,
                min_y=min_y,
                room_w=room_w,
                room_h=room_h,
                cell_size_px=cell_size_px,
            )
            sx_cad, sy_cad = pixel_to_cad(
                px=spx,
                py=spy,
                Xmin=cad_params["Xmin"],
                Ymin=cad_params["Ymin"],
                Xmax=cad_params["Xmax"],
                Ymax=cad_params["Ymax"],
                width=image_width,
                height=image_height,
            )
            switch_info = {
                "switch_type": "开关",
                "grid_position": [srow, scol],
                "pixel_position": [float(spx), float(spy)],
                "cad_position": [float(sx_cad), float(sy_cad)],
            }

        compact_room = {
            "room_name": room_name,
            "lamp_count": int(len(lamp_grid_positions)),
            "room_area_m2": float(base_internal.get("room_area_m2", 0.0)),
            "switch": {
                "switch_type": str(switch_info.get("switch_type", "开关")),
                "cad_position": list(switch_info.get("cad_position", [])),
            } if switch_info else None,
            "switch_count": 1 if switch_info else 0,
            "lamps": {
                "lamp_type": lamp_type,
                "count": int(len(lamp_cad_positions)),
                "cad_positions": lamp_cad_positions,
            },
        }

        internal_room = copy.deepcopy(base_internal)
        internal_room.update(
            {
                "room_name": room_name,
                "lamp_count": int(len(lamp_grid_positions)),
                "placement_mode": placement_mode,
                "target_lux": float(target_lux),
                "required_flux_per_lamp_lm": float(required_flux_per_lamp_lm) if required_flux_per_lamp_lm is not None else None,
                "lamp_model": lamp_model,
                "switch": switch_info,
                "switch_count": 1 if switch_info else 0,
                "switches": [switch_info] if switch_info else [],
                "lamps": {
                    "lamp_type": lamp_type,
                    "lamp_model": lamp_model,
                    "count": int(len(lamp_grid_positions)),
                    "grid_positions": lamp_grid_positions,
                    "pixel_positions": lamp_pixel_positions,
                    "cad_positions": lamp_cad_positions,
                },
            }
        )
        return compact_room, internal_room

    def _build_room_wiring_plan(
        self,
        *,
        room_name: str,
        room_plan: Optional[Dict[str, Any]],
    ) -> RoomWiringPlan:
        layer_name = os.getenv("CAD_WIRING_LAYER_NAME", "wiringlayer")
        color = os.getenv("CAD_WIRING_COLOR", "yellow")
        try:
            line_width = float(os.getenv("CAD_WIRING_LINE_WIDTH", "20"))
        except Exception:
            line_width = 20.0

        payload = room_plan if isinstance(room_plan, dict) else {}
        status = str(payload.get("status", "ok"))
        segments: List[WiringSegment] = []
        merged_segments = payload.get("merged_segments_cad", []) or []
        if isinstance(merged_segments, list):
            for seg in merged_segments:
                if not isinstance(seg, list) or len(seg) != 2:
                    continue
                p1, p2 = seg[0], seg[1]
                if (
                    not isinstance(p1, list)
                    or not isinstance(p2, list)
                    or len(p1) < 2
                    or len(p2) < 2
                ):
                    continue
                try:
                    start = [float(p1[0]), float(p1[1]), 0.0]
                    end = [float(p2[0]), float(p2[1]), 0.0]
                except Exception:
                    continue
                segments.append(
                    WiringSegment(
                        start_cad=start,
                        end_cad=end,
                        layer_name=layer_name,
                        line_width=line_width,
                        color=color,
                    )
                )

        return RoomWiringPlan(
            room_name=str(payload.get("room_name", room_name)),
            status=status,
            segment_count=len(segments),
            segments=segments,
        )

    def _summarize_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        summarized: Dict[str, Any] = {}
        for key, value in (constraints or {}).items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            summarized[key] = value
        return summarized

    def _choose_chat_start_from(
        self,
        intent_type: str,
        constraints: Dict[str, Any],
        existing_state: Optional[Dict[str, Any]],
    ) -> str:
        if intent_type == "rerun_wiring":
            return "wiring"
        if constraints.get("lamps") is not None or constraints.get("switch") is not None:
            return "validation"
        if constraints.get("lamp_count") is not None:
            return "placement"
        if constraints.get("target_lux") is not None or constraints.get("required_flux_per_lamp_lm") is not None:
            return "count"
        if constraints.get("lamp_type") is not None or constraints.get("lamp_model") is not None:
            return "model"
        if existing_state:
            return "validation"
        return "requirement"

    def _build_agent_user_goal(
        self,
        *,
        intent_type: str,
        room_name: str,
        constraints: Dict[str, Any],
    ) -> str:
        lines = [
            f"目标房间: {room_name}",
            f"任务类型: {intent_type}",
            "请在当前房间矩阵上完成照明设计与布线。",
        ]

        if intent_type == "rerun_wiring":
            lines.append("保持当前灯具和开关位置不变，只重新生成布线。")

        explicit_constraints = self._summarize_constraints(constraints)
        if explicit_constraints:
            lines.append("必须遵守以下显式约束：")
            for key, value in explicit_constraints.items():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _seed_agent_state_from_existing(
        self,
        *,
        state: RoomAgentState,
        existing_state: Optional[Dict[str, Any]],
    ) -> None:
        if not existing_state:
            return

        matrix = np.array(state.matrix, dtype=np.int32)
        lamps_raw = existing_state.get("lamps")
        if lamps_raw is None:
            lighting_internal = existing_state.get("lighting_internal") or {}
            lamps_raw = ((lighting_internal.get("lamps", {}) or {}).get("grid_positions")) or []
        try:
            lamps = self._normalize_grid_points(lamps_raw, matrix, allow_door_cells=False)
            state.placements["lamps"] = [[int(row), int(col)] for row, col in lamps]
        except Exception:
            state.placements["lamps"] = []

        switches: List[List[int]] = []
        switch_obj = existing_state.get("switch")
        if not isinstance(switch_obj, dict):
            lighting_internal = existing_state.get("lighting_internal") or {}
            switch_obj = lighting_internal.get("switch") or {}
        if isinstance(switch_obj, dict) and switch_obj.get("grid_position") is not None:
            try:
                normalized_switch = self._normalize_grid_points(
                    [switch_obj.get("grid_position")],
                    matrix,
                    allow_door_cells=True,
                )
                switches = [[int(normalized_switch[0][0]), int(normalized_switch[0][1])]]
            except Exception:
                switches = []
        state.placements["switches"] = switches
        state.selected_lamp_type = (existing_state.get("selected_lamp_type") or None)
        cached_plan = ((existing_state.get("tool_cache") or {}).get("lamp_plan"))
        if isinstance(cached_plan, dict):
            state.lamp_plan = copy.deepcopy(cached_plan)
            state.tool_cache["lamp_plan"] = copy.deepcopy(cached_plan)

    def _build_agent_trace(
        self,
        *,
        state: RoomAgentState,
        finish_reason: str,
        strategy_summary: str,
    ) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = []
        used_thought_indexes = set()
        saw_finish = False
        for idx, tool_entry in enumerate(state.tool_history, 1):
            tool_name = str(tool_entry.get("tool", "")) if isinstance(tool_entry, dict) else ""
            if tool_name == "finish":
                saw_finish = True
            thought_text = ""
            for thought_idx, thought_item in enumerate(state.thought_history):
                if thought_idx in used_thought_indexes:
                    continue
                action_name = str(thought_item.get("action") or "")
                if action_name == tool_name:
                    thought_text = str(thought_item.get("thought") or "")
                    used_thought_indexes.add(thought_idx)
                    break
            trace.append(
                self._trace_entry(
                    idx,
                    thought_text,
                    tool_name,
                    tool_entry.get("input", {}) if isinstance(tool_entry, dict) else {},
                    str(tool_entry.get("tool_result", "")),
                )
            )

        if not saw_finish:
            trace.append(
                self._trace_entry(
                    len(trace) + 1,
                    "",
                    "finish",
                    {"reason": finish_reason},
                    strategy_summary or finish_reason,
                )
            )
        return trace

    def save_uploaded_files(self, files: List[bytes], filenames: List[str], target_dir: Optional[Path] = None) -> tuple[str, List[str]]:
        """
        保存上传的文件到临时目录
        :param files: 文件内容列表
        :param filenames: 文件名列表
        :return: (临时目录路径, 保存的文件路径列表)
        """
        if target_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="cad_upload_")
            write_dir = Path(temp_dir)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = str(target_dir)
            write_dir = target_dir
        saved_files = []
        
        try:
            for file_content, filename in zip(files, filenames):
                # 确保文件名安全
                safe_filename = Path(filename).name
                file_path = str(write_dir / safe_filename)
                
                # 写入文件
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                
                saved_files.append(file_path)
                logger.debug(f"保存上传文件: {file_path}")
            
            return temp_dir, saved_files
            
        except Exception as e:
            # 出错时清理已创建的文件
            self.cleanup_temp_directory(temp_dir)
            raise Exception(f"保存上传文件失败: {str(e)}")
    
    def cleanup_temp_directory(self, temp_dir: str):
        """
        清理临时目录
        :param temp_dir: 临时目录路径
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"清理临时目录: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时目录失败 {temp_dir}: {str(e)}")
    
    def process_uploaded_files(
        self,
        files: List[bytes],
        filenames: List[str],
        cad_params: Optional[CADParams] = None,
        run_layout: bool = False,
    ) -> CADResponse:
        """
        处理上传的文件
        :param files: 文件内容列表
        :param filenames: 文件名列表
        :param cad_params: CAD参数
        :return: CAD分析响应
        """
        logger.info(f"开始处理上传的文件，共 {len(files)} 个文件")
        
        # 验证上传的文件
        is_valid, error_msg = self.validate_uploaded_files(files, filenames)
        if not is_valid:
            logger.warning(f"文件验证失败: {error_msg}")
            return CADResponse(
                success=False,
                message=f"文件验证失败: {error_msg}",
                total_images=0,
                processed_images=0,
                results={},
                lighting_results={},
                wiring_results={},
                errors={"validation": error_msg}
            )
        
        temp_dir = None
        session_id = None
        session_dir: Optional[Path] = None
        try:
            session_id, session_dir = self._create_session_workspace()
            uploads_dir = session_dir / "uploads"
            temp_dir, saved_files = self.save_uploaded_files(files, filenames, target_dir=uploads_dir)
            
            # 转换CAD参数
            cad_params_dict = self.convert_cad_params(cad_params)
            logger.debug(f"使用CAD参数: {cad_params_dict}")
            
            # 执行批量处理
            logger.info("开始批量处理上传的图像...")
            ## 开始处理图片
            processing_results = process_images_batch(
                image_directory=temp_dir,
                cad_params=cad_params_dict,
                save_to_file=True,  # 服务模式下保存文件
                run_layout=run_layout,
            )
            
            # 统计处理结果
            total_images = len(processing_results)
            error_results = {k: v['error'] for k, v in processing_results.items() 
                           if 'error' in v}
            processed_images = total_images - len(error_results)
            
            logger.info(f"批量处理完成: {processed_images}/{total_images} 个图像成功")
            
            if error_results:
                logger.warning(f"处理失败的图像: {list(error_results.keys())}")
            
            # 提取房间坐标
            room_coordinates = self.extract_room_coordinates(processing_results)
            lighting_results = self.extract_lighting_results(processing_results)
            wiring_results = self.extract_wiring_results(processing_results)
            
            # 构建响应
            session_images: Dict[str, Any] = {}
            for image_name, result in processing_results.items():
                image_path = str(uploads_dir / image_name)
                image_payload: Dict[str, Any] = {"image_path": image_path}
                if isinstance(result, dict) and "error" not in result:
                    image_payload["intermediate"] = {
                        "cad_rooms": result.get("cad_rooms", {}),
                        "room_rectangles": result.get("room_rectangles", {}),
                        "door_assignments": result.get("door_assignments", []),
                    }
                    if run_layout:
                        image_payload["lighting_rooms"] = result.get("lighting_rooms", {})
                        image_payload["wiring_rooms"] = result.get("wiring_rooms", {})
                else:
                    image_payload["error"] = result.get("error") if isinstance(result, dict) else str(result)
                session_images[image_name] = image_payload

            self._write_session_manifest(
                session_id,
                {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "cad_params": cad_params_dict,
                    "images": session_images,
                },
            )

            if processed_images > 0:
                response = CADResponse(
                    success=True,
                    message=f"成功处理 {processed_images}/{total_images} 个图像",
                    session_id=session_id,
                    total_images=total_images,
                    processed_images=processed_images,
                    results=room_coordinates,
                    lighting_results=lighting_results if run_layout else {},
                    wiring_results=wiring_results if run_layout else {},
                    errors=error_results
                )
                logger.info(f"CAD处理成功: 提取了 {sum(len(rooms) for rooms in room_coordinates.values())} 个房间的坐标")
            else:
                response = CADResponse(
                    success=False,
                    message="所有图像处理均失败",
                    session_id=session_id,
                    total_images=total_images,
                    processed_images=0,
                    results={},
                    lighting_results={},
                    wiring_results={},
                    errors=error_results
                )
                logger.error("CAD处理失败: 所有图像处理均失败")
            
            return response
            
        except Exception as e:
            logger.error(f"处理上传文件时发生异常: {str(e)}")
            logger.error(f"异常详情: {e.__class__.__name__}")
            
            return CADResponse(
                success=False,
                message=f"处理过程中发生错误: {str(e)}",
                session_id=session_id,
                total_images=0,
                processed_images=0,
                results={},
                lighting_results={},
                wiring_results={},
                errors={"processing": str(e)}
            )
        
        finally:
            if temp_dir and run_layout and session_dir is None:
                self.cleanup_temp_directory(temp_dir)

    def process_all_rooms(self, session_id: str, placement_mode: Optional[str] = None) -> CADLayoutResponse:
        """
        基于 upload-and-process 生成的中间结果，执行步骤7和步骤8。
        """
        resolved_mode = (placement_mode or "").strip().lower()
        if resolved_mode and resolved_mode not in {"llm", "rule"}:
            return CADLayoutResponse(
                success=False,
                message=f"无效的 placement_mode: {placement_mode}",
                session_id=session_id,
                total_images=0,
                processed_images=0,
                lighting_results={},
                wiring_results={},
                errors={"placement_mode": f"invalid placement_mode: {placement_mode}"},
            )

        logger.info(
            f"开始处理会话的步骤7和步骤8: session_id={session_id}, "
            f"placement_mode={resolved_mode or 'env/default'}"
        )

        try:
            manifest = self._read_session_manifest(session_id)
        except Exception as exc:
            logger.warning(f"读取会话失败: {exc}")
            return CADLayoutResponse(
                success=False,
                message=f"读取会话失败: {exc}",
                session_id=session_id,
                total_images=0,
                processed_images=0,
                lighting_results={},
                wiring_results={},
                errors={"session": str(exc)},
            )

        cad_params_dict = manifest.get("cad_params") or self.default_cad_params
        images = manifest.get("images") or {}
        if not isinstance(images, dict) or not images:
            return CADLayoutResponse(
                success=False,
                message="会话中没有可处理的图像",
                session_id=session_id,
                total_images=0,
                processed_images=0,
                lighting_results={},
                wiring_results={},
                errors={"session": "会话中没有可处理的图像"},
            )

        processing_results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        total_images = len(images)
        processed_images = 0
        session_dir = SESSION_ROOT / session_id
        layouts_root = session_dir / "layouts"
        layouts_root.mkdir(parents=True, exist_ok=True)

        for image_name, image_payload in images.items():
            try:
                if not isinstance(image_payload, dict):
                    raise ValueError("图像会话数据格式错误")
                if image_payload.get("error"):
                    raise ValueError(str(image_payload.get("error")))

                image_path = str(image_payload.get("image_path") or "")
                intermediate = image_payload.get("intermediate") or {}
                room_rectangles = intermediate.get("room_rectangles") or {}
                door_assignments = intermediate.get("door_assignments") or []
                if not image_path or not room_rectangles:
                    raise ValueError("缺少步骤1-6的中间结果")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                per_image_output_dir = layouts_root / f"{Path(image_name).stem}_{timestamp}"
                layout_result = process_layout_from_intermediate(
                    image_path=image_path,
                    room_rectangles=room_rectangles,
                    door_assignments=door_assignments,
                    cad_params=cad_params_dict,
                    placement_mode=resolved_mode or None,
                    save_to_file=True,
                    output_dir=str(per_image_output_dir),
                )

                image_payload["lighting_rooms"] = layout_result.get("lighting_rooms", {})
                image_payload["wiring_rooms"] = layout_result.get("wiring_rooms", {})
                processing_results[image_name] = {
                    "lighting_rooms": image_payload["lighting_rooms"],
                    "wiring_rooms": image_payload["wiring_rooms"],
                }
                processed_images += 1
            except Exception as exc:
                logger.error(f"会话 {session_id} 的图像 {image_name} 执行步骤7/8失败: {exc}")
                errors[image_name] = str(exc)
                processing_results[image_name] = {"error": str(exc)}

        manifest["images"] = images
        self._write_session_manifest(session_id, manifest)

        lighting_results = self.extract_lighting_results(processing_results)
        wiring_results = self.extract_wiring_results(processing_results)

        success = processed_images > 0
        message = (
            f"成功处理 {processed_images}/{total_images} 个图像的步骤7和步骤8"
            if success
            else "所有图像的步骤7和步骤8处理均失败"
        )
        return CADLayoutResponse(
            success=success,
            message=message,
            session_id=session_id,
            total_images=total_images,
            processed_images=processed_images,
            lighting_results=lighting_results,
            wiring_results=wiring_results,
            errors=errors,
        )

    def get_chat_state(
        self,
        session_id: str,
        conversation_id: Optional[str] = None,
        room_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        manifest = self._read_session_manifest(session_id)
        response: Dict[str, Any] = {
            "success": True,
            "session_id": session_id,
            "available_rooms": self._list_available_rooms(manifest),
            "available_conversations": self._list_conversation_ids(session_id),
        }

        if conversation_id is None or not str(conversation_id).strip():
            return response

        resolved_id, conversation = self._load_conversation(session_id, conversation_id, create=False)
        response["conversation_id"] = resolved_id
        response["conversation"] = conversation

        if room_name:
            response["room_state"] = self._load_room_state(session_id, resolved_id, room_name)
        else:
            response["room_states"] = self._load_all_room_states(session_id, resolved_id)
        return response

    def reset_chat_room(
        self,
        session_id: str,
        conversation_id: str,
        room_name: str,
    ) -> Dict[str, Any]:
        resolved_id, conversation = self._load_conversation(session_id, conversation_id, create=False)
        room_path = self._room_state_path(session_id, resolved_id, room_name)
        removed = False
        if room_path.exists():
            room_path.unlink()
            removed = True

        conversation["updated_at"] = datetime.now().isoformat()
        if conversation.get("current_room") == room_name:
            conversation["current_room"] = None
        history = conversation.get("history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "intent_type": "reset_room",
                "room_name": room_name,
                "observation": "removed" if removed else "no_state",
            }
        )
        conversation["history"] = history[-50:]
        self._write_json_file(self._conversation_path(session_id, resolved_id), conversation)

        return {
            "success": True,
            "message": "房间状态已重置" if removed else "房间状态不存在，无需重置",
            "session_id": session_id,
            "conversation_id": resolved_id,
            "room_name": room_name,
            "removed": removed,
        }

    def process_chat_turn(
        self,
        *,
        session_id: str,
        conversation_id: Optional[str],
        intent_type: str,
        room_name: Optional[str],
        constraints: Optional[Dict[str, Any]] = None,
        execution: Optional[Dict[str, Any]] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        resolved_intent = (intent_type or "").strip()
        if not resolved_intent:
            raise ValueError("intent_type is required")

        constraints = dict(constraints or {})
        execution = dict(execution or {})
        manifest = self._read_session_manifest(session_id)

        if resolved_intent == "list_rooms":
            response = self.get_chat_state(session_id, conversation_id=conversation_id)
            response["intent_type"] = resolved_intent
            return response

        if resolved_intent == "get_room_state":
            if not conversation_id:
                raise ValueError("conversation_id is required for get_room_state")
            response = self.get_chat_state(session_id, conversation_id=conversation_id, room_name=room_name)
            response["intent_type"] = resolved_intent
            return response

        if resolved_intent == "reset_room":
            if not conversation_id:
                raise ValueError("conversation_id is required for reset_room")
            if not room_name:
                raise ValueError("room_name is required for reset_room")
            response = self.reset_chat_room(session_id, conversation_id, room_name)
            response["intent_type"] = resolved_intent
            return response

        allowed_intents = {"design_room", "update_room", "rerun_layout", "rerun_wiring"}
        if resolved_intent not in allowed_intents:
            raise ValueError(f"unsupported intent_type: {resolved_intent}")

        resolved_id, conversation = self._load_conversation(session_id, conversation_id, create=True)
        target_room = (room_name or conversation.get("current_room") or "").strip()
        if not target_room:
            raise ValueError("room_name is required")
        stream_event_sink: Optional[Callable[[Dict[str, Any]], None]] = None
        if event_sink is not None:
            def stream_event_sink(event: Dict[str, Any]) -> None:
                enriched = dict(event or {})
                enriched.setdefault("session_id", session_id)
                enriched.setdefault("conversation_id", resolved_id)
                enriched.setdefault("room_name", target_room)
                enriched.setdefault("intent_type", resolved_intent)
                event_sink(enriched)

        room_context = self._resolve_room_context(manifest, target_room)
        existing_state = self._load_room_state(session_id, resolved_id, target_room)
        explicit_constraints = self._summarize_constraints(constraints)

        explicit_placement_mode = constraints.get("placement_mode")
        if explicit_placement_mode is not None:
            explicit_placement_mode = str(explicit_placement_mode).strip().lower()
            if explicit_placement_mode not in {"rule", "llm"}:
                raise ValueError("placement_mode must be 'rule' or 'llm'")

        switch_count = constraints.get("switch_count")
        if switch_count is not None:
            switch_count = int(switch_count)
            if switch_count not in {0, 1}:
                raise ValueError("当前仅支持0或1个开关")

        execution_start = str(execution.get("start_from", "auto") or "auto").strip().lower()
        valid_start_from = {"auto", "requirement", "model", "count", "placement", "validation", "wiring"}
        if execution_start not in valid_start_from:
            raise ValueError(f"unsupported start_from: {execution_start}")
        start_from = (
            self._choose_chat_start_from(resolved_intent, constraints, existing_state)
            if execution_start == "auto"
            else execution_start
        )

        reuse_existing = bool(execution.get("resume_existing", True))
        overwrite_existing = bool(execution.get("overwrite_existing", False))
        run_wiring = bool(constraints.get("run_wiring", True))

        resolved_placement_mode = (
            explicit_placement_mode
            or (str((existing_state or {}).get("placement_mode", "")).strip().lower() if existing_state else "")
            or str(os.getenv("CAD_LIGHTING_PLACEMENT_MODE", "rule")).strip().lower()
        )
        if resolved_placement_mode not in {"rule", "llm"}:
            resolved_placement_mode = "rule"

        trace: List[Dict[str, Any]] = []
        trace.append(
            self._trace_entry(
                1,
                "先解析本轮执行边界，确保显式约束优先于历史状态。",
                "resolve_chat_turn",
                {
                    "intent_type": resolved_intent,
                    "room_name": target_room,
                    "conversation_id": resolved_id,
                    "explicit_constraints": explicit_constraints,
                    "execution": execution,
                },
                f"start_from={start_from}, placement_mode={resolved_placement_mode}, run_wiring={run_wiring}",
            )
        )
        self._emit_chat_event(
            stream_event_sink,
            "status",
            {
                "phase": "resolve_chat_turn",
                "start_from": start_from,
                "placement_mode": resolved_placement_mode,
                "run_wiring": run_wiring,
                "explicit_constraints": explicit_constraints,
            },
        )

        base_internal = None
        base_compact = None
        image_width = 0
        image_height = 0
        if (
            resolved_intent == "rerun_wiring"
            and reuse_existing
            and not overwrite_existing
            and existing_state
            and isinstance(existing_state.get("lighting_internal"), dict)
        ):
            base_internal = copy.deepcopy(existing_state.get("lighting_internal") or {})
            base_compact = copy.deepcopy(existing_state.get("lighting_result") or {})
            image_width = int((existing_state.get("tool_cache") or {}).get("image_width", 0))
            image_height = int((existing_state.get("tool_cache") or {}).get("image_height", 0))
            trace.append(
                self._trace_entry(
                    2,
                    "本轮只重跑布线，优先复用已有的房间离散化和灯位状态。",
                    "reuse_room_state",
                    {
                        "lamp_count": existing_state.get("lamp_count"),
                        "placement_mode": existing_state.get("placement_mode"),
                    },
                    "loaded lighting_internal from existing room state",
                )
            )

            if image_width <= 1 or image_height <= 1:
                base_internal = None
                base_compact = None

        if not isinstance(base_internal, dict) or not base_internal:
            lighting_payload = process_room_lighting_layout(
                room_rectangles={target_room: room_context["room_shapes"]},
                image_path=room_context["image_path"],
                cad_params=room_context["cad_params"],
                door_assignments=room_context.get("door_assignments") or [],
                placement_mode=resolved_placement_mode,
                save_to_file=False,
            )
            base_internal = (lighting_payload.get("rooms_internal") or {}).get(target_room) or {}
            base_compact = (lighting_payload.get("rooms") or {}).get(target_room) or {}
            image_width = int(lighting_payload.get("image_width", 0))
            image_height = int(lighting_payload.get("image_height", 0))
            if not base_internal:
                raise ValueError(f"房间离散化失败: {target_room}")
            trace.append(
                self._trace_entry(
                    2,
                    "需要可执行的房间基础网格和默认灯位，因此先对单房间重建步骤7基础状态。",
                    "process_room_lighting_layout",
                    {
                        "image_name": room_context["image_name"],
                        "room_name": target_room,
                        "placement_mode": resolved_placement_mode,
                    },
                    (
                        f"room_area_m2={float(base_internal.get('room_area_m2', 0.0)):.2f}, "
                        f"estimated_lamp_count={int(base_internal.get('estimated_lamp_count', 0))}"
                    ),
                )
            )
        self._emit_chat_event(
            stream_event_sink,
            "status",
            {
                "phase": "base_layout_ready",
                "room_area_m2": float(base_internal.get("room_area_m2", 0.0)),
                "estimated_lamp_count": int(base_internal.get("estimated_lamp_count", 0) or 0),
                "image_name": room_context["image_name"],
            },
        )

        room_area_m2 = float(base_internal.get("room_area_m2", 0.0))
        inherited_target_lux = (existing_state or {}).get("target_lux")
        if inherited_target_lux is None:
            inherited_target_lux = _get_target_lux_for_room(target_room)
        target_lux = float(
            constraints.get("target_lux")
            if constraints.get("target_lux") is not None
            else inherited_target_lux
        )

        selected_lamp_type = str(
            constraints.get("lamp_type")
            or (existing_state or {}).get("selected_lamp_type")
            or ((base_compact or {}).get("lamps", {}) or {}).get("lamp_type")
            or "筒灯"
        ).strip()
        lamp_model = (
            str(constraints.get("lamp_model")).strip()
            if constraints.get("lamp_model") is not None
            else ((existing_state or {}).get("lamp_model"))
        )
        catalog_lamp = self._resolve_catalog_lamp(selected_lamp_type, lamp_model)
        lamp_flux_lm = catalog_lamp.get("phi_lm")

        if constraints.get("lamps") is not None and constraints.get("lamp_count") is not None:
            if len(constraints.get("lamps") or []) != int(constraints.get("lamp_count")):
                raise ValueError("lamp_count 必须与 lamps 提供的坐标数量一致")

        if constraints.get("lamps") is not None:
            resolved_lamp_count = len(constraints.get("lamps") or [])
            lamp_count_source = "explicit_lamps"
        elif constraints.get("lamp_count") is not None:
            resolved_lamp_count = int(constraints.get("lamp_count"))
            lamp_count_source = "explicit_count"
        elif constraints.get("target_lux") is not None and lamp_flux_lm:
            resolved_lamp_count = max(
                1,
                int(np.ceil((target_lux * room_area_m2) / (0.6 * 0.8 * float(lamp_flux_lm)))),
            )
            lamp_count_source = "target_lux"
        elif reuse_existing and not overwrite_existing and existing_state and existing_state.get("lamp_count") is not None:
            resolved_lamp_count = int(existing_state.get("lamp_count"))
            lamp_count_source = "existing_state"
        else:
            resolved_lamp_count = int(
                ((base_compact or {}).get("lamp_count"))
                or base_internal.get("estimated_lamp_count")
                or 1
            )
            lamp_count_source = "auto"

        if resolved_lamp_count <= 0:
            raise ValueError("lamp_count must be greater than 0")

        required_flux_per_lamp_lm = (
            float(constraints.get("required_flux_per_lamp_lm"))
            if constraints.get("required_flux_per_lamp_lm") is not None
            else self._calc_required_flux_per_lamp(
                area_m2=room_area_m2,
                target_lux=target_lux,
                lamp_count=resolved_lamp_count,
            )
        )

        resolved_switch_count = switch_count
        if resolved_switch_count is None:
            if (existing_state or {}).get("switch") is not None:
                resolved_switch_count = 1
            elif base_internal.get("switch") is not None or (base_internal.get("door_edge_cells") or []):
                resolved_switch_count = 1
            else:
                resolved_switch_count = 0

        has_explicit_plan_change = any(
            constraints.get(key) is not None
            for key in (
                "lamp_type",
                "lamp_count",
                "target_lux",
                "required_flux_per_lamp_lm",
                "lamp_model",
                "placement_mode",
                "switch_count",
            )
        )
        reset_layout = bool(
            existing_state is None
            or overwrite_existing
            or resolved_intent in {"design_room", "rerun_layout"}
            or (resolved_intent == "update_room" and has_explicit_plan_change)
        )
        skip_initial_design = bool(resolved_intent == "rerun_wiring" and existing_state is not None)
        if skip_initial_design:
            reset_layout = False

        agent_state = RoomAgentState(
            room_name=target_room,
            area_m2=room_area_m2,
            matrix=np.array(base_internal.get("matrix") or [], dtype=np.int32),
        )
        if reuse_existing and not overwrite_existing:
            self._seed_agent_state_from_existing(state=agent_state, existing_state=existing_state)

        agent_plan_overrides = {
            "lamp_type": selected_lamp_type,
            "lamp_count": resolved_lamp_count,
            "target_lux": int(target_lux),
            "switch_count": int(resolved_switch_count),
            "is_regular": bool(base_internal.get("is_regular", True)),
        }
        from agent.factory import build_lighting_agent

        agent = build_lighting_agent(
            provider=os.getenv("CAD_AGENT_PROVIDER", "qwen").strip().lower(),
            model_name=os.getenv("CAD_AGENT_MODEL", "").strip() or None,
            init_mode=resolved_placement_mode,
            log_dir=str(self._conversation_dir(session_id, resolved_id) / "logs"),
            event_sink=stream_event_sink,
        )
        self._emit_chat_event(
            stream_event_sink,
            "status",
            {
                "phase": "agent_run_start",
                "reset_layout": reset_layout,
                "skip_initial_design": skip_initial_design,
                "placement_mode": resolved_placement_mode,
                "lamp_count": resolved_lamp_count,
                "switch_count": resolved_switch_count,
            },
        )
        agent_result = agent.run_for_room(
            state=agent_state,
            max_steps=max(1, int(os.getenv("CAD_AGENT_MAX_STEPS", "6"))),
            user_goal=self._build_agent_user_goal(
                intent_type=resolved_intent,
                room_name=target_room,
                constraints=constraints,
            ),
            reset_layout=reset_layout,
            plan_overrides=agent_plan_overrides,
            skip_initial_design=skip_initial_design,
            generate_wiring=False,
        )

        trace_notes: List[Dict[str, Any]] = []
        matrix = np.array(agent_state.matrix, dtype=np.int32)

        if constraints.get("lamps") is not None:
            explicit_lamps = self._normalize_grid_points(
                constraints.get("lamps"),
                matrix,
                allow_door_cells=False,
            )
            agent_state.placements["lamps"] = [[int(row), int(col)] for row, col in explicit_lamps]
            trace_notes.append(
                self._trace_entry(
                    0,
                    "客户端显式给出了灯具坐标，因此覆盖 agent 自动布置结果。",
                    "apply_explicit_lamps",
                    {"lamps": constraints.get("lamps")},
                    f"lamp_count={len(explicit_lamps)}",
                )
            )
        else:
            try:
                current_lamps = self._normalize_grid_points(
                    agent_state.placements.get("lamps", []),
                    matrix,
                    allow_door_cells=False,
                )
            except Exception:
                current_lamps = []

            if len(current_lamps) != resolved_lamp_count:
                if resolved_placement_mode == "rule":
                    if bool(base_internal.get("is_regular")):
                        adjusted_lamps = _select_lamp_cells_regular_grid(matrix, resolved_lamp_count)
                        adjust_source = "rule_regular"
                    else:
                        adjusted_lamps = _select_lamp_cells_rule_based(matrix, resolved_lamp_count)
                        adjust_source = "rule_irregular"
                else:
                    adjusted_lamps = _complete_lamp_cells(current_lamps, matrix, resolved_lamp_count)
                    adjust_source = "agent_complete"
                agent_state.placements["lamps"] = [[int(row), int(col)] for row, col in adjusted_lamps]
                trace_notes.append(
                    self._trace_entry(
                        0,
                        "为了满足显式数量约束，对 agent 结果做了最小化后处理。",
                        "normalize_lamp_count",
                        {"target_lamp_count": resolved_lamp_count},
                        f"adjust_source={adjust_source}",
                    )
                )

        if resolved_switch_count == 0:
            if agent_state.placements.get("switches"):
                trace_notes.append(
                    self._trace_entry(
                        0,
                        "客户端显式要求不放置开关，因此清空 agent 产生的开关。",
                        "clear_switches",
                        {"switch_count": 0},
                        "switches cleared",
                    )
                )
            agent_state.placements["switches"] = []
        elif constraints.get("switch") is not None:
            explicit_switch = self._normalize_grid_points(
                [constraints.get("switch")],
                matrix,
                allow_door_cells=True,
            )[0]
            agent_state.placements["switches"] = [[int(explicit_switch[0]), int(explicit_switch[1])]]
            trace_notes.append(
                self._trace_entry(
                    0,
                    "客户端显式给出了开关坐标，因此覆盖 agent 自动布置结果。",
                    "apply_explicit_switch",
                    {"switch": constraints.get("switch")},
                    f"switch={list(explicit_switch)}",
                )
            )
        else:
            normalized_switches: List[Tuple[int, int]] = []
            for item in agent_state.placements.get("switches", []):
                try:
                    normalized_switch = self._normalize_grid_points([item], matrix, allow_door_cells=True)[0]
                    normalized_switches.append(normalized_switch)
                except Exception:
                    continue

            if not normalized_switches and resolved_switch_count > 0:
                fallback_switch, _ = self._select_switch_for_chat(
                    base_internal,
                    explicit_switch=None,
                    existing_state=existing_state,
                    preserve_existing=bool(reuse_existing and not overwrite_existing),
                )
                if fallback_switch is not None:
                    normalized_switches = [fallback_switch]
                    trace_notes.append(
                        self._trace_entry(
                            0,
                            "agent 未给出可用开关位置，因此回退到靠门规则位置。",
                            "fallback_switch",
                            {"target_switch_count": resolved_switch_count},
                            f"switch={list(fallback_switch)}",
                        )
                    )

            if len(normalized_switches) > 1:
                normalized_switches = normalized_switches[:1]
            agent_state.placements["switches"] = [[int(row), int(col)] for row, col in normalized_switches]

        final_validation = agent.tools.tool_validate_layout(state=agent_state)

        final_lamp_type = str(
            constraints.get("lamp_type")
            or ((agent_result.get("lamp_plan", {}) or {}).get("selected_lamp", {}) or {}).get("lamp_type")
            or agent_result.get("selected_lamp_type")
            or agent_state.selected_lamp_type
            or selected_lamp_type
        ).strip()
        final_lamp_model = (
            str(constraints.get("lamp_model")).strip()
            if constraints.get("lamp_model") is not None
            else (((agent_result.get("lamp_plan", {}) or {}).get("selected_lamp", {}) or {}).get("model") or lamp_model)
        )
        final_catalog_lamp = self._resolve_catalog_lamp(final_lamp_type, final_lamp_model)
        agent_state.selected_lamp_type = final_lamp_type
        if not isinstance(agent_state.lamp_plan, dict):
            agent_state.lamp_plan = {}
        selected_lamp_payload = dict(((agent_state.lamp_plan or {}).get("selected_lamp", {}) or {}))
        selected_lamp_payload["lamp_type"] = final_lamp_type
        if final_lamp_model:
            selected_lamp_payload["model"] = final_lamp_model
        agent_state.lamp_plan.update(
            {
                "room_name": target_room,
                "target_lux": int(target_lux),
                "lamp_count": int(len(agent_state.placements.get("lamps", []))),
                "switch_count": int(len(agent_state.placements.get("switches", []))),
                "required_flux_per_lamp_lm": float(required_flux_per_lamp_lm) if required_flux_per_lamp_lm is not None else None,
                "selected_lamp": selected_lamp_payload,
            }
        )
        agent_state.tool_cache["lamp_plan"] = copy.deepcopy(agent_state.lamp_plan)

        lamp_cells = self._normalize_grid_points(
            agent_state.placements.get("lamps", []),
            matrix,
            allow_door_cells=False,
        )
        switch_cell = None
        if agent_state.placements.get("switches"):
            switch_cell = self._normalize_grid_points(
                [agent_state.placements.get("switches", [])[0]],
                matrix,
                allow_door_cells=True,
            )[0]

        compact_room, internal_room = self._build_room_lighting_from_cells(
            room_name=target_room,
            base_internal=base_internal,
            image_width=image_width,
            image_height=image_height,
            cad_params=room_context["cad_params"],
            lamp_type=final_lamp_type,
            lamp_model=final_lamp_model,
            target_lux=target_lux,
            required_flux_per_lamp_lm=required_flux_per_lamp_lm,
            placement_mode=resolved_placement_mode,
            lamp_cells=lamp_cells,
            switch_cell=switch_cell,
        )

        if run_wiring:
            self._emit_chat_event(
                stream_event_sink,
                "status",
                {
                    "phase": "wiring_start",
                    "lamp_count": len(agent_state.placements.get("lamps", [])),
                },
            )
            wiring_room_raw = agent.tools.tool_generate_wiring(
                state=agent_state,
                bbox_pixel=base_internal.get("bbox_pixel"),
                cell_size_px=int(base_internal.get("cell_size_px", 40)),
                cad_params=room_context["cad_params"],
                image_width=image_width,
                image_height=image_height,
            )
        else:
            wiring_room_raw = {
                "room_name": target_room,
                "status": "skipped",
                "reason": "run_wiring_false",
                "route_count": 0,
                "routes": [],
            }
            trace_notes.append(
                self._trace_entry(
                    0,
                    "本轮只更新房间设计，不执行布线。",
                    "skip_wiring",
                    {"run_wiring": False},
                    "wiring skipped by constraint",
                )
            )
            self._emit_chat_event(
                stream_event_sink,
                "status",
                {
                    "phase": "wiring_skipped",
                },
            )
        wiring_room = self._build_room_wiring_plan(
            room_name=target_room,
            room_plan=wiring_room_raw,
        ).model_dump()

        trace.extend(
            self._build_agent_trace(
                state=agent_state,
                finish_reason=str(agent_result.get("finish_reason", "done")),
                strategy_summary=str(agent_result.get("strategy_summary", "")),
            )
        )
        if trace and trace[-1].get("action") == "finish":
            finish_entry = trace.pop()
        else:
            finish_entry = None
        for note in trace_notes:
            note["step"] = len(trace) + 1
            trace.append(note)
        if finish_entry is not None:
            finish_entry["step"] = len(trace) + 1
            trace.append(finish_entry)
        for idx, entry in enumerate(trace, 1):
            entry["step"] = idx

        validation = {
            "agent_validation": final_validation,
            "lamp_count_matches_positions": int(compact_room.get("lamp_count", 0)) == len(((internal_room.get("lamps", {}) or {}).get("grid_positions", [])) or []),
            "switch_count_supported": compact_room.get("switch_count", 0) in (0, 1),
            "run_wiring": run_wiring,
        }

        timestamp = datetime.now().isoformat()
        execution_record = {
            "timestamp": timestamp,
            "intent_type": resolved_intent,
            "start_from": start_from,
            "constraints": explicit_constraints,
            "result": {
                "lamp_type": final_lamp_type,
                "lamp_count": compact_room.get("lamp_count", 0),
                "wiring_status": wiring_room.get("status", "unknown"),
            },
        }
        execution_history = []
        if existing_state and isinstance(existing_state.get("execution_history"), list):
            execution_history.extend(existing_state.get("execution_history") or [])
        execution_history.append(execution_record)

        room_state = {
            "session_id": session_id,
            "conversation_id": resolved_id,
            "room_name": target_room,
            "image_name": room_context["image_name"],
            "image_path": room_context["image_path"],
            "selected_lamp_type": final_lamp_type,
            "lamp_count": int(compact_room.get("lamp_count", 0)),
            "target_lux": target_lux,
            "required_flux_per_lamp_lm": required_flux_per_lamp_lm,
            "lamp_model": final_lamp_model,
            "placement_mode": resolved_placement_mode,
            "lamps": ((internal_room.get("lamps", {}) or {}).get("grid_positions")) or [],
            "switch": internal_room.get("switch"),
            "lighting_result": compact_room,
            "lighting_internal": internal_room,
            "wiring": wiring_room,
            "validation": validation,
            "tool_cache": {
                "image_width": image_width,
                "image_height": image_height,
                "room_area_m2": room_area_m2,
                "estimated_lamp_count": base_internal.get("estimated_lamp_count"),
                "catalog_lamp": final_catalog_lamp,
                "lamp_plan": copy.deepcopy(agent_state.lamp_plan),
                "lamp_model_plan": copy.deepcopy(agent_state.tool_cache.get("lamp_model_plan")),
                "component_count_plan": copy.deepcopy(agent_state.tool_cache.get("component_count_plan")),
                "agent_validation": copy.deepcopy(final_validation),
                "agent_finish_reason": agent_result.get("finish_reason"),
                "agent_strategy_summary": agent_result.get("strategy_summary"),
                "agent_log_file": agent_result.get("log_file"),
            },
            "execution_history": execution_history[-50:],
            "updated_at": timestamp,
        }
        self._write_json_file(self._room_state_path(session_id, resolved_id, target_room), room_state)

        history = conversation.get("history", [])
        if not isinstance(history, list):
            history = []
        history.append(execution_record)
        conversation.update(
            {
                "current_room": target_room,
                "turn_count": int(conversation.get("turn_count", 0)) + 1,
                "updated_at": timestamp,
                "history": history[-50:],
            }
        )
        global_preferences = conversation.get("global_preferences")
        if not isinstance(global_preferences, dict):
            global_preferences = {}
        global_preferences["placement_mode"] = resolved_placement_mode
        conversation["global_preferences"] = global_preferences
        self._write_json_file(self._conversation_path(session_id, resolved_id), conversation)
        self._emit_chat_event(
            stream_event_sink,
            "status",
            {
                "phase": "persisted",
                "lamp_count": int(compact_room.get("lamp_count", 0)),
                "wiring_status": wiring_room.get("status", "unknown"),
            },
        )

        return {
            "success": True,
            "message": "chat turn processed",
            "session_id": session_id,
            "conversation_id": resolved_id,
            "intent_type": resolved_intent,
            "room_name": target_room,
            "image_name": room_context["image_name"],
            "start_from": start_from,
            "lighting_result": compact_room,
            "wiring_result": wiring_room,
            "room_state": room_state,
            "trace": trace,
        }
    
    def convert_cad_params(self, cad_params: Optional[CADParams]) -> dict:
        """
        转换CAD参数为处理函数需要的格式
        :param cad_params: CAD参数对象
        :return: CAD参数字典
        """
        if cad_params is None:
            return self.default_cad_params
        
        return {
            'Xmin': cad_params.Xmin,
            'Ymin': cad_params.Ymin,
            'Xmax': cad_params.Xmax,
            'Ymax': cad_params.Ymax,
        }
    
    def extract_room_coordinates(self, processing_results: Dict[str, Any]) -> Dict[str, List[RoomCADCoordinate]]:
        """
        从处理结果中提取房间CAD坐标
        :param processing_results: 批量处理结果
        :return: 格式化的房间坐标数据
        """
        formatted_results = {}
        
        for image_name, result in processing_results.items():
            if isinstance(result, dict) and 'error' in result:
                # 跳过处理失败的图像
                continue
            
            image_rooms = []
            
            # 处理结果可能直接就是cad_rooms字典，或者包含cad_rooms字段
            if isinstance(result, dict) and 'cad_rooms' in result:
                cad_rooms = result['cad_rooms']
            elif isinstance(result, dict):
                # 直接就是cad_rooms字典
                cad_rooms = result
            else:
                # 其他情况，跳过
                continue
            
            for room_name, room_coordinates in cad_rooms.items():
                room_coord = RoomCADCoordinate(
                    room_name=room_name,
                    cad_coordinates=room_coordinates
                )
                image_rooms.append(room_coord)
            
            formatted_results[image_name] = image_rooms
        
        return formatted_results

    def extract_lighting_results(self, processing_results: Dict[str, Any]) -> Dict[str, List[RoomLightingPlan]]:
        """
        从处理结果中提取灯具布置结果。
        :param processing_results: 批量处理结果
        :return: 格式化的房间灯具结果
        """
        formatted_results: Dict[str, List[RoomLightingPlan]] = {}

        for image_name, result in processing_results.items():
            if isinstance(result, dict) and "error" in result:
                continue

            image_rooms: List[RoomLightingPlan] = []
            lighting_rooms = {}
            if isinstance(result, dict) and "lighting_rooms" in result:
                lighting_rooms = result.get("lighting_rooms", {}) or {}

            if not isinstance(lighting_rooms, dict):
                formatted_results[image_name] = image_rooms
                continue

            for room_name, room_plan in lighting_rooms.items():
                if not isinstance(room_plan, dict):
                    continue

                # 兼容新旧格式:
                # 新格式: lamps = {lamp_type,count,grid_positions,pixel_positions,cad_positions}
                # 旧格式: lamps = [{lamp_type,grid_position,pixel_position,cad_position}, ...]
                lamps_raw = room_plan.get("lamps", {}) or {}
                if isinstance(lamps_raw, dict):
                    lamps_plan = LampsPlan(
                        lamp_type=str(lamps_raw.get("lamp_type", "筒灯")),
                        count=int(lamps_raw.get("count", room_plan.get("lamp_count", 0))),
                        cad_positions=[list(x) for x in lamps_raw.get("cad_positions", []) or []],
                    )
                else:
                    old_list = lamps_raw if isinstance(lamps_raw, list) else []
                    cad_positions = [list(x.get("cad_position", [0.0, 0.0])) for x in old_list if isinstance(x, dict)]
                    lamp_type = "筒灯"
                    for x in old_list:
                        if isinstance(x, dict) and x.get("lamp_type"):
                            lamp_type = str(x.get("lamp_type"))
                            break
                    lamps_plan = LampsPlan(
                        lamp_type=lamp_type,
                        count=int(len(cad_positions)),
                        cad_positions=cad_positions,
                    )

                switch_obj: Optional[SwitchPlacement] = None
                sw = room_plan.get("switch")
                if isinstance(sw, dict):
                    switch_obj = SwitchPlacement(
                        switch_type=str(sw.get("switch_type", "开关")),
                        cad_position=list(sw.get("cad_position", [])),
                    )
                else:
                    switches_raw = room_plan.get("switches", []) or []
                    if isinstance(switches_raw, list) and switches_raw:
                        sw0 = switches_raw[0]
                        if isinstance(sw0, dict):
                            switch_obj = SwitchPlacement(
                                switch_type=str(sw0.get("switch_type", "开关")),
                                cad_position=list(sw0.get("cad_position", [])),
                            )

                image_rooms.append(
                    RoomLightingPlan(
                        room_name=str(room_plan.get("room_name", room_name)),
                        lamp_count=int(room_plan.get("lamp_count", lamps_plan.count)),
                        room_area_m2=float(room_plan.get("room_area_m2", 0.0)),
                        lamps=lamps_plan,
                        switch=switch_obj,
                        switch_count=int(room_plan.get("switch_count", 1 if switch_obj else 0)),
                    )
                )

            formatted_results[image_name] = image_rooms

        return formatted_results

    def extract_wiring_results(self, processing_results: Dict[str, Any]) -> Dict[str, List[RoomWiringPlan]]:
        """
        从处理结果中提取步骤8布线结果。
        输出每条线段的两个端点以及CAD绘制参数。
        """
        formatted_results: Dict[str, List[RoomWiringPlan]] = {}

        for image_name, result in processing_results.items():
            if isinstance(result, dict) and "error" in result:
                continue

            image_rooms: List[RoomWiringPlan] = []
            wiring_rooms = {}
            if isinstance(result, dict) and "wiring_rooms" in result:
                wiring_rooms = result.get("wiring_rooms", {}) or {}

            if not isinstance(wiring_rooms, dict):
                formatted_results[image_name] = image_rooms
                continue

            for room_name, room_plan in wiring_rooms.items():
                if not isinstance(room_plan, dict):
                    continue
                image_rooms.append(
                    self._build_room_wiring_plan(
                        room_name=room_name,
                        room_plan=room_plan,
                    )
                )

            formatted_results[image_name] = image_rooms

        return formatted_results
    


# 全局服务实例
cad_service = CADAnalysisService()
