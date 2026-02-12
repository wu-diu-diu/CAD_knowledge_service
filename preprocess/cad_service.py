"""
CAD分析服务主逻辑模块
处理外部请求，调用核心分析功能，返回房间CAD坐标
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
import os
import tempfile
import shutil
from pathlib import Path
from .find_all import process_images_batch, process_single_image
from .coordinate_converter import DEFAULT_CAD_PARAMS
from .logger import get_logger

# 获取logger实例
logger = get_logger("cad_service")


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


class CADResponse(BaseModel):
    """CAD分析响应数据模型"""
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="处理结果消息")
    total_images: int = Field(..., description="处理的图片总数")
    processed_images: int = Field(..., description="成功处理的图片数")
    results: Dict[str, List[RoomCADCoordinate]] = Field(..., description="每个图片的房间CAD坐标结果")
    errors: Dict[str, str] = Field(default_factory=dict, description="处理错误信息")


class CADAnalysisService:
    """CAD分析服务类"""
    
    def __init__(self):
        self.default_cad_params = DEFAULT_CAD_PARAMS
    
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
    
    def save_uploaded_files(self, files: List[bytes], filenames: List[str]) -> tuple[str, List[str]]:
        """
        保存上传的文件到临时目录
        :param files: 文件内容列表
        :param filenames: 文件名列表
        :return: (临时目录路径, 保存的文件路径列表)
        """
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="cad_upload_")
        saved_files = []
        
        try:
            for file_content, filename in zip(files, filenames):
                # 确保文件名安全
                safe_filename = Path(filename).name
                file_path = os.path.join(temp_dir, safe_filename)
                
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
    
    def process_uploaded_files(self, files: List[bytes], filenames: List[str], 
                             cad_params: Optional[CADParams] = None) -> CADResponse:
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
                errors={"validation": error_msg}
            )
        
        temp_dir = None
        try:
            # 保存上传的文件到临时目录
            temp_dir, saved_files = self.save_uploaded_files(files, filenames)
            
            # 转换CAD参数
            cad_params_dict = self.convert_cad_params(cad_params)
            logger.debug(f"使用CAD参数: {cad_params_dict}")
            
            # 执行批量处理
            logger.info("开始批量处理上传的图像...")
            processing_results = process_images_batch(
                image_directory=temp_dir,
                cad_params=cad_params_dict,
                save_to_file=True  # 服务模式下保存文件
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
            
            # 构建响应
            if processed_images > 0:
                response = CADResponse(
                    success=True,
                    message=f"成功处理 {processed_images}/{total_images} 个图像",
                    total_images=total_images,
                    processed_images=processed_images,
                    results=room_coordinates,
                    errors=error_results
                )
                logger.info(f"CAD处理成功: 提取了 {sum(len(rooms) for rooms in room_coordinates.values())} 个房间的坐标")
            else:
                response = CADResponse(
                    success=False,
                    message="所有图像处理均失败",
                    total_images=total_images,
                    processed_images=0,
                    results={},
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
                total_images=0,
                processed_images=0,
                results={},
                errors={"processing": str(e)}
            )
        
        finally:
            # 清理临时目录
            if temp_dir:
                self.cleanup_temp_directory(temp_dir)
    
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
    
    def process_cad_request(self, request: CADRequest) -> CADResponse:
        """
        处理CAD分析请求
        :param request: CAD分析请求
        :return: CAD分析响应
        """
        logger.info(f"开始处理CAD请求: 目录={request.image_directory}")
        
        # 验证请求
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            logger.warning(f"请求验证失败: {error_msg}")
            return CADResponse(
                success=False,
                message=f"请求验证失败: {error_msg}",
                total_images=0,
                processed_images=0,
                results={},
                errors={"validation": error_msg}
            )
        
        try:
            # 转换CAD参数
            cad_params = self.convert_cad_params(request.cad_params)
            logger.debug(f"使用CAD参数: {cad_params}")
            
            # 执行批量处理
            logger.info("开始批量处理图像...")
            processing_results = process_images_batch(
                image_directory=request.image_directory,
                cad_params=cad_params,
                save_to_file=False  # 服务模式下不保存文件
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
            
            # 构建响应
            if processed_images > 0:
                response = CADResponse(
                    success=True,
                    message=f"成功处理 {processed_images}/{total_images} 个图像",
                    total_images=total_images,
                    processed_images=processed_images,
                    results=room_coordinates,
                    errors=error_results
                )
                logger.info(f"CAD处理成功: 提取了 {sum(len(rooms) for rooms in room_coordinates.values())} 个房间的坐标")
            else:
                response = CADResponse(
                    success=False,
                    message="所有图像处理均失败",
                    total_images=total_images,
                    processed_images=0,
                    results={},
                    errors=error_results
                )
                logger.error("CAD处理失败: 所有图像处理均失败")
            
            return response
            
        except Exception as e:
            logger.error(f"处理CAD请求时发生异常: {str(e)}")
            logger.error(f"异常详情: {e.__class__.__name__}")
            
            return CADResponse(
                success=False,
                message=f"处理过程中发生错误: {str(e)}",
                total_images=0,
                processed_images=0,
                results={},
                errors={"processing": str(e)}
            )


# 全局服务实例
cad_service = CADAnalysisService()