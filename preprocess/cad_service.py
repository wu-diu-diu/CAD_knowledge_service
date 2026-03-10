"""
CAD分析服务主逻辑模块
处理外部请求，调用核心分析功能，返回房间CAD坐标
"""
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
import json
import os
import tempfile
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from .find_all import process_images_batch, process_single_image, process_layout_from_intermediate
from .coordinate_converter import DEFAULT_CAD_PARAMS
from .logger import get_logger

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

        layer_name = os.getenv("CAD_WIRING_LAYER_NAME", "wiringlayer")
        color = os.getenv("CAD_WIRING_COLOR", "yellow")
        try:
            line_width = float(os.getenv("CAD_WIRING_LINE_WIDTH", "20"))
        except Exception:
            line_width = 20.0

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

                status = str(room_plan.get("status", "ok"))
                segments: List[WiringSegment] = []

                merged_segments = room_plan.get("merged_segments_cad", []) or []
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

                image_rooms.append(
                    RoomWiringPlan(
                        room_name=str(room_plan.get("room_name", room_name)),
                        status=status,
                        segment_count=len(segments),
                        segments=segments,
                    )
                )

            formatted_results[image_name] = image_rooms

        return formatted_results
    


# 全局服务实例
cad_service = CADAnalysisService()
