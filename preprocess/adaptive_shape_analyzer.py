"""
自适应房间形状分析模块
提供多种形状表示方法来处理不规则房间形状
"""
import cv2
import numpy as np
import math
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt


def calculate_shape_complexity(points):
    """
    计算形状复杂度指标
    :param points: 房间轮廓点列表 [[x1,y1], [x2,y2], ...]
    :return: 复杂度分数 (0-1, 越高越复杂)
    """
    if len(points) < 3:
        return 0.0
    
    points_array = np.array(points, dtype=np.float32)
    
    # 1. 计算凸包
    hull = cv2.convexHull(points_array)
    
    # 2. 计算面积比 (轮廓面积 / 凸包面积)
    contour_area = cv2.contourArea(points_array)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return 0.0
    
    area_ratio = contour_area / hull_area
    
    # 3. 计算周长比 (轮廓周长 / 凸包周长)  
    contour_perimeter = cv2.arcLength(points_array, True)
    hull_perimeter = cv2.arcLength(hull, True)
    
    if hull_perimeter == 0:
        return 0.0
        
    perimeter_ratio = contour_perimeter / hull_perimeter
    
    # 4. 计算转折点密度 (点数 / 凸包周长)
    point_density = len(points) / hull_perimeter if hull_perimeter > 0 else 0
    
    # 综合复杂度分数 (面积比越小、周长比越大、点密度越高 = 越复杂)
    complexity = (1 - area_ratio) * 0.4 + (perimeter_ratio - 1) * 0.4 + min(point_density / 0.1, 1.0) * 0.2
    
    return float(min(max(complexity, 0.0), 1.0))


def calculate_convex_hull(points):
    """
    计算凸包作为形状的外边界表示
    :param points: 房间轮廓点列表
    :return: 凸包点列表 [[x,y], ...]
    """
    if len(points) < 3:
        return points
    
    points_array = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points_array)
    
    # 转换为列表格式
    hull_points = []
    for point in hull:
        hull_points.append([int(point[0][0]), int(point[0][1])])
    
    return hull_points


def calculate_oriented_bounding_box(points):
    """
    计算定向边界框(旋转矩形)，比轴对齐矩形更紧密
    :param points: 房间轮廓点列表
    :return: 定向边界框的四个角点 [[x,y], ...]
    """
    if len(points) < 3:
        return points
    
    points_array = np.array(points, dtype=np.float32)
    
    # 计算最小面积旋转矩形
    rect = cv2.minAreaRect(points_array)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # 按逆时针顺序排序角点
    center = np.mean(box, axis=0)
    angles = []
    for point in box:
        angle = math.atan2(point[1] - center[1], point[0] - center[0])
        angles.append(angle)
    
    # 按角度排序
    sorted_indices = sorted(range(len(angles)), key=lambda i: angles[i])
    sorted_box = [box[i] for i in sorted_indices]
    
    # 转换为整数列表
    result = []
    for point in sorted_box:
        result.append([int(point[0]), int(point[1])])
    
    return result


def calculate_alpha_shape(points, alpha=0.1):
    """
    计算Alpha形状，提供更精确的边界表示
    注：这是简化版本，使用凸包作为近似
    :param points: 房间轮廓点列表  
    :param alpha: Alpha参数
    :return: Alpha形状点列表
    """
    # 简化实现：对于复杂形状使用凸包
    return calculate_convex_hull(points)


def decompose_complex_polygon(points, max_sides=8):
    """
    将复杂多边形分解为更简单的几何形状
    :param points: 房间轮廓点列表
    :param max_sides: 简化后的最大边数
    :return: 简化的多边形点列表
    """
    if len(points) <= max_sides:
        return points
    
    points_array = np.array(points, dtype=np.float32)
    
    # 使用Douglas-Peucker算法简化多边形
    epsilon = 0.01 * cv2.arcLength(points_array, True)
    approx = cv2.approxPolyDP(points_array, epsilon, True)
    
    # 如果仍然太复杂，增加epsilon
    while len(approx) > max_sides and epsilon < 0.1 * cv2.arcLength(points_array, True):
        epsilon *= 1.5
        approx = cv2.approxPolyDP(points_array, epsilon, True)
    
    # 转换为列表格式
    result = []
    for point in approx:
        result.append([int(point[0][0]), int(point[0][1])])
    
    return result


def choose_best_shape_representation(points):
    """
    根据形状复杂度选择最适合的表示方法
    :param points: 房间轮廓点列表
    :return: 字典包含不同表示方法和推荐方法
    """
    complexity = calculate_shape_complexity(points)
    
    results = {
        'complexity': complexity,
        'axis_aligned_bbox': None,
        'oriented_bbox': None, 
        'convex_hull': None,
        'alpha_shape': None,
        'simplified_polygon': None,
        'recommended': None
    }
    
    # 计算轴对齐边界框 (原方法)
    if len(points) >= 4:
        points_array = np.array(points, dtype=np.float32)
        rect = cv2.minAreaRect(points_array)
        box = cv2.boxPoints(rect)
        
        # 转换为轴对齐矩形
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        results['axis_aligned_bbox'] = [
            [min_x, min_y], [max_x, min_y], 
            [max_x, max_y], [min_x, max_y]
        ]
    
    # 计算各种表示方法
    results['oriented_bbox'] = calculate_oriented_bounding_box(points)
    results['convex_hull'] = calculate_convex_hull(points)
    results['alpha_shape'] = calculate_alpha_shape(points)
    results['simplified_polygon'] = decompose_complex_polygon(points)
    
    # 根据复杂度选择推荐方法
    if complexity < 0.1:
        # 简单形状：使用轴对齐边界框
        results['recommended'] = 'axis_aligned_bbox'
    elif complexity < 0.3:
        # 中等复杂：使用定向边界框
        results['recommended'] = 'oriented_bbox'  
    elif complexity < 0.6:
        # 较复杂：使用凸包
        results['recommended'] = 'convex_hull'
    else:
        # 非常复杂：使用简化多边形
        results['recommended'] = 'simplified_polygon'
    
    return results


def analyze_all_room_shapes(processed_rooms):
    """
    分析所有房间的形状并提供适应性表示
    :param processed_rooms: 处理后的房间数据
    :return: 形状分析结果字典
    """
    analysis_results = {}
    
    print("\n=== 房间形状复杂度分析 ===")
    
    for room_name, room_contours in processed_rooms.items():
        if not room_contours or len(room_contours) == 0:
            continue
            
        # 取第一个轮廓进行分析
        contour = room_contours[0]
        if len(contour) < 3:
            continue
            
        # 分析形状
        shape_analysis = choose_best_shape_representation(contour)
        analysis_results[room_name] = shape_analysis
        
        complexity = shape_analysis['complexity']
        recommended = shape_analysis['recommended']
        
        print(f"{room_name}: 复杂度={complexity:.3f}, 推荐方法={recommended}")
    
    return analysis_results


def visualize_shape_representations(room_name, shape_analysis, image_path, save_output=True):
    """
    可视化不同的形状表示方法
    :param room_name: 房间名称
    :param shape_analysis: 形状分析结果
    :param image_path: 原始图像路径
    :param save_output: 是否保存结果
    """
    # 读取原始图像
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{room_name} - 不同形状表示方法对比', fontsize=16)
    
    methods = [
        ('axis_aligned_bbox', '轴对齐边界框'),
        ('oriented_bbox', '定向边界框'), 
        ('convex_hull', '凸包'),
        ('alpha_shape', 'Alpha形状'),
        ('simplified_polygon', '简化多边形'),
        ('recommended', f"推荐方法: {shape_analysis['recommended']}")
    ]
    
    for i, (method_key, title) in enumerate(methods):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # 显示原始图像
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 获取对应的形状点
        if method_key == 'recommended':
            shape_points = shape_analysis[shape_analysis['recommended']]
        else:
            shape_points = shape_analysis.get(method_key)
        
        if shape_points and len(shape_points) > 2:
            # 绘制形状
            points_array = np.array(shape_points + [shape_points[0]], dtype=np.int32)  # 闭合多边形
            x_coords = points_array[:, 0]
            y_coords = points_array[:, 1]
            ax.plot(x_coords, y_coords, 'r-', linewidth=2)
            ax.fill(x_coords, y_coords, alpha=0.3, color='red')
        
        ax.set_title(f'{title}\n复杂度: {shape_analysis["complexity"]:.3f}')
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
    
    plt.tight_layout()
    
    if save_output:
        output_dir = 'images/output'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'shape_analysis_{room_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"形状分析结果已保存: {output_file}")
    
    plt.show()


def process_adaptive_room_shapes(processed_rooms, image_path, save_to_file=True):
    """
    处理所有房间的自适应形状表示
    :param processed_rooms: 处理后的房间数据
    :param image_path: 图像路径
    :param save_to_file: 是否保存到文件
    :return: 自适应形状分析结果
    """
    print("\n=== 开始自适应形状分析 ===")
    
    # 分析所有房间形状
    analysis_results = analyze_all_room_shapes(processed_rooms)
    
    # 生成改进的房间表示
    improved_rooms = {}
    
    for room_name, shape_analysis in analysis_results.items():
        recommended_method = shape_analysis['recommended']
        recommended_shape = shape_analysis[recommended_method]
        
        improved_rooms[room_name] = {
            'original_points': processed_rooms[room_name][0] if processed_rooms[room_name] else [],
            'complexity': shape_analysis['complexity'],
            'recommended_method': recommended_method,
            'recommended_shape': recommended_shape,
            'all_methods': {
                'axis_aligned_bbox': shape_analysis['axis_aligned_bbox'],
                'oriented_bbox': shape_analysis['oriented_bbox'],
                'convex_hull': shape_analysis['convex_hull'], 
                'simplified_polygon': shape_analysis['simplified_polygon']
            }
        }
    
    # 保存结果
    if save_to_file:
        output_dir = 'images/output'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'adaptive_room_shapes.json')
        
        # 转换numpy类型为Python类型以支持JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        serializable_data = convert_numpy_types(improved_rooms)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"自适应形状分析结果已保存: {output_file}")
    
    print("=== 自适应形状分析完成 ===\n")
    
    return improved_rooms


def create_comparison_visualization(improved_rooms, image_path):
    """
    创建原始边界框与改进方法的对比可视化
    :param improved_rooms: 改进的房间形状数据
    :param image_path: 图像路径
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左图：原始轴对齐边界框
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('原始方法：轴对齐边界框', fontsize=14)
    
    # 右图：自适应推荐方法
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.set_title('改进方法：自适应形状表示', fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(improved_rooms)))
    
    for i, (room_name, room_data) in enumerate(improved_rooms.items()):
        color = colors[i]
        
        # 绘制原始边界框 (左图)
        if room_data['all_methods']['axis_aligned_bbox']:
            bbox = room_data['all_methods']['axis_aligned_bbox']
            bbox_array = np.array(bbox + [bbox[0]], dtype=np.int32)
            ax1.plot(bbox_array[:, 0], bbox_array[:, 1], color=color, linewidth=2, label=room_name)
            ax1.fill(bbox_array[:, 0], bbox_array[:, 1], alpha=0.2, color=color)
        
        # 绘制推荐方法 (右图)
        if room_data['recommended_shape']:
            shape = room_data['recommended_shape']
            shape_array = np.array(shape + [shape[0]], dtype=np.int32)
            method_name = room_data['recommended_method']
            ax2.plot(shape_array[:, 0], shape_array[:, 1], color=color, linewidth=2, 
                    label=f'{room_name} ({method_name})')
            ax2.fill(shape_array[:, 0], shape_array[:, 1], alpha=0.2, color=color)
    
    # 设置图例和坐标轴
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for ax in [ax1, ax2]:
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
    
    plt.tight_layout()
    
    # 保存对比图
    output_dir = 'images/output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'shape_method_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"方法对比图已保存: {output_file}")
    
    plt.show()
