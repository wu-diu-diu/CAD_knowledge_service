"""
坐标转换模块
将PNG像素坐标转换为CAD坐标系统
"""
import cv2
import json
import os
import numpy as np


def _to_builtin_types(obj):
    """
    将numpy类型递归转换为可JSON序列化的Python内建类型。
    """
    if isinstance(obj, np.ndarray):
        return _to_builtin_types(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _to_builtin_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin_types(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def pixel_to_cad(px, py, Xmin, Ymin, Xmax, Ymax, width, height):
    """
    将像素坐标转换为CAD坐标
    :param px: 像素x坐标
    :param py: 像素y坐标  
    :param Xmin: CAD窗口左边界
    :param Ymin: CAD窗口下边界
    :param Xmax: CAD窗口右边界
    :param Ymax: CAD窗口上边界
    :param width: PNG图像宽度
    :param height: PNG图像高度
    :return: CAD坐标 (x_cad, y_cad)
    """
    if width <= 1 or height <= 1:
        raise ValueError(f"无效图像尺寸: width={width}, height={height}")
    if Xmax == Xmin or Ymax == Ymin:
        raise ValueError(
            f"无效CAD范围: X({Xmin}, {Xmax}), Y({Ymin}, {Ymax})"
        )

    # 像素坐标边界保护，避免越界点导致外推
    px = max(0.0, min(float(px), float(width - 1)))
    py = max(0.0, min(float(py), float(height - 1)))

    # 使用像素端点精确映射:
    # Xcad = xmin + (px / (W - 1)) * (xmax - xmin)
    # Ycad = ymax - (py / (H - 1)) * (ymax - ymin)
    x_cad = Xmin + (px / (width - 1)) * (Xmax - Xmin)
    y_cad = Ymax - (py / (height - 1)) * (Ymax - Ymin)
    return x_cad, y_cad


def convert_rec_to_cad(processed_rooms, cad_params, image_path, return_image_size=False):
    """
    将processed_rooms中所有房间的坐标点坐标转换为CAD坐标
    :param processed_rooms: 处理后的房间最小外接矩形字典
    :param cad_params: CAD参数字典，至少包含 Xmin/Ymin/Xmax/Ymax
    :param image_path: PNG图像路径（用于获取尺寸）
    :param return_image_size: 是否同时返回图像宽高
    :return: 转换后的CAD坐标字典，或(cad_rooms, width, height)
    """
    # 读取图像获取尺寸
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    
    height, width = img.shape[:2]
    print(f"图像尺寸：宽={width}, 高={height}")
    
    # 提取CAD参数
    Xmin = cad_params['Xmin']
    Ymin = cad_params['Ymin']
    Xmax = cad_params['Xmax']
    Ymax = cad_params['Ymax']
    print(f"CAD窗口范围: X({Xmin:.4f}, {Xmax:.4f}), Y({Ymin:.4f}, {Ymax:.4f})")
    if 'originx' in cad_params or 'originy' in cad_params:
        print("提示: 当前坐标转换不再使用originx/originy，按四角线性映射进行转换。")
    
    # 转换所有房间的轮廓点
    cad_rooms = {}
    
    def _is_point(item):
        if isinstance(item, np.ndarray):
            return item.ndim == 1 and item.shape[0] == 2
        return (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], (int, float, np.integer, np.floating))
            and isinstance(item[1], (int, float, np.integer, np.floating))
        )

    def _normalize_shapes(room_shapes):
        """
        兼容两种结构：
        1) 单个多边形: [[x,y], [x,y], ...]
        2) 多个多边形: [ [[x,y],...], [[x,y],...], ... ]
        """
        if room_shapes is None:
            return []
        if isinstance(room_shapes, np.ndarray):
            if room_shapes.ndim == 2 and room_shapes.shape[1] == 2:
                return [room_shapes.tolist()]
            if room_shapes.ndim == 3 and room_shapes.shape[2] == 2:
                return [shape.tolist() for shape in room_shapes]
        if not isinstance(room_shapes, (list, tuple)) or len(room_shapes) == 0:
            return []
        first = room_shapes[0]
        if _is_point(first):
            return [room_shapes]
        if isinstance(first, np.ndarray) and first.ndim == 2 and first.shape[1] == 2:
            return [shape.tolist() if isinstance(shape, np.ndarray) else shape for shape in room_shapes]
        if isinstance(first, (list, tuple)) and len(first) > 0 and _is_point(first[0]):
            return list(room_shapes)
        return []

    for room_name, room_shapes in processed_rooms.items():
        cad_rooms[room_name] = []
        polygons = _normalize_shapes(room_shapes)
        for polygon in polygons:
            cad_contour = []
            for rec_points in polygon:
                px, py = rec_points[0], rec_points[1]
                x_cad, y_cad = pixel_to_cad(
                    px, py, Xmin, Ymin, Xmax, Ymax, width, height
                )
                cad_contour.append([x_cad, y_cad])
            if cad_contour:
                cad_rooms[room_name].append(cad_contour)
    
    if return_image_size:
        return cad_rooms, width, height
    return cad_rooms


def save_cad_coordinates(cad_rooms, output_file, image_width=None, image_height=None):
    """
    保存CAD坐标到JSON文件
    :param cad_rooms: CAD坐标字典
    :param output_file: 输出文件路径
    :param image_width: 图像宽度（可选）
    :param image_height: 图像高度（可选）
    """
    serializable_rooms = _to_builtin_types(cad_rooms)

    serializable_data = {
        "image_width": image_width,
        "image_height": image_height,
        "rooms": serializable_rooms,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
    
    print(f"CAD坐标已保存到: {output_file}")


def save_pixel_coordinates(room_rectangles, output_file, image_width=None, image_height=None):
    """
    保存步骤6转换前(像素坐标)的房间轮廓数据到JSON文件
    :param room_rectangles: 房间像素坐标字典
    :param output_file: 输出文件路径
    :param image_width: 图像宽度（可选）
    :param image_height: 图像高度（可选）
    """
    serializable_rooms = _to_builtin_types(room_rectangles)

    serializable_data = {
        "image_width": image_width,
        "image_height": image_height,
        "rooms": serializable_rooms,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    print(f"像素坐标已保存到: {output_file}")


def create_cad_params(Xmin, Ymin, Xmax, Ymax, originx=0, originy=0):
    """
    创建CAD参数字典
    :param Xmin: CAD窗口左边界
    :param Ymin: CAD窗口下边界
    :param Xmax: CAD窗口右边界
    :param Ymax: CAD窗口上边界
    :param originx: X轴原点偏移量
    :param originy: Y轴原点偏移量
    :return: CAD参数字典
    """
    return {
        'Xmin': Xmin,
        'Ymin': Ymin,
        'Xmax': Xmax,
        'Ymax': Ymax,
    }

# 示例：默认CAD参数（基于现有的transfor_png_to_cad.py）
DEFAULT_CAD_PARAMS = create_cad_params(
    Xmin=18231.64, 
    Ymin=24612.05,
    Xmax=43352.90, 
    Ymax=69272.05,
)

def process_rooms_to_cad(room_rectangles, image_path, cad_params=None, save_to_file=True):
    """
    将处理后的房间外接矩形转换为CAD坐标并保存
    :param room_rectangles: 房间的最小外接矩形的坐标
    :param image_path: PNG图像路径
    :param cad_params: CAD参数字典，如果为None则使用默认参数
    :param save_to_file: 是否保存到文件
    :return: CAD坐标字典
    """
    if cad_params is None:
        cad_params = DEFAULT_CAD_PARAMS
        print("使用默认CAD参数")
    
    print("\n=== 开始坐标转换 ===")
    
    # 转换坐标
    cad_rooms, image_width, image_height = convert_rec_to_cad(
        room_rectangles, cad_params, image_path, return_image_size=True
    )
    
    # 保存到文件
    if save_to_file:
        output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
        os.makedirs(output_dir, exist_ok=True)
        pixel_output_file = os.path.join(output_dir, 'room_contours_pixels.json')
        save_pixel_coordinates(
            room_rectangles,
            pixel_output_file,
            image_width=image_width,
            image_height=image_height,
        )

        output_file = os.path.join(output_dir, 'room_contours_cad.json')
        save_cad_coordinates(
            cad_rooms,
            output_file,
            image_width=image_width,
            image_height=image_height,
        )
    
    print("=== 坐标转换完成 ===\n")
    
    return cad_rooms
