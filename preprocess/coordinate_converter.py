"""
坐标转换模块
将PNG像素坐标转换为CAD坐标系统
"""
import cv2
import json
import os
import numpy as np


def pixel_to_cad(px, py, Xmin, Ymin, Xmax, Ymax, width, height, originx, originy):
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
    :param originx: X轴原点偏移量
    :param originy: Y轴原点偏移量
    :return: CAD坐标 (x_cad, y_cad)
    """
    x_cad = Xmin + (px + originx) / width * (Xmax - Xmin)
    y_cad = Ymin + (height - py - originy) / height * (Ymax - Ymin)
    return x_cad, y_cad


def convert_rec_to_cad(processed_rooms, cad_params, image_path):
    """
    将processed_rooms中所有房间的坐标点坐标转换为CAD坐标
    :param processed_rooms: 处理后的房间最小外接矩形字典
    :param cad_params: CAD参数字典，包含窗口数据和原点偏移
    :param image_path: PNG图像路径（用于获取尺寸）
    :return: 转换后的CAD坐标字典
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
    originx = cad_params['originx']
    originy = cad_params['originy']
    
    print(f"CAD窗口范围: X({Xmin:.4f}, {Xmax:.4f}), Y({Ymin:.4f}, {Ymax:.4f})")
    print(f"原点偏移: ({originx}, {originy})")
    
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
                    px, py, Xmin, Ymin, Xmax, Ymax,
                    width, height, originx, originy
                )
                cad_contour.append([x_cad, y_cad])
            if cad_contour:
                cad_rooms[room_name].append(cad_contour)
    
    return cad_rooms


def save_cad_coordinates(cad_rooms, output_file):
    """
    保存CAD坐标到JSON文件
    :param cad_rooms: CAD坐标字典
    :param output_file: 输出文件路径
    """
    # 转换numpy数组为普通列表以便JSON序列化
    serializable_data = {}
    for room_name, contour_list in cad_rooms.items():
        serializable_data[room_name] = []
        for contour in contour_list:
            serializable_data[room_name].append(contour)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
    
    print(f"CAD坐标已保存到: {output_file}")


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
        'originx': originx,
        'originy': originy
    }

# 示例：默认CAD参数（基于现有的transfor_png_to_cad.py）
DEFAULT_CAD_PARAMS = create_cad_params(
    Xmin=10543.9662, 
    Ymin=59282.8141,
    Xmax=21881.1076, 
    Ymax=69995.4501,
    originx=11, 
    originy=-13
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
    cad_rooms = convert_rec_to_cad(room_rectangles, cad_params, image_path)
    
    # 保存到文件
    if save_to_file:
        output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'room_contours_cad.json')
        save_cad_coordinates(cad_rooms, output_file)
    
    print("=== 坐标转换完成 ===\n")
    
    return cad_rooms
