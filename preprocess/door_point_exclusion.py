"""
门内转折点删除模块
负责从房间轮廓中排除位于门区域内的转折点
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def _snap_axis(values, tolerance):
    """
    将同一轴上彼此接近的坐标对齐到同一值（曼哈顿化的一部分）。
    """
    if len(values) == 0:
        return np.array([], dtype=np.int32)

    unique_vals = sorted(set(int(v) for v in values))
    groups = []
    current_group = [unique_vals[0]]
    for val in unique_vals[1:]:
        if abs(val - current_group[-1]) <= tolerance:
            current_group.append(val)
        else:
            groups.append(current_group)
            current_group = [val]
    groups.append(current_group)

    value_to_center = {}
    for group in groups:
        center = int(round(sum(group) / len(group)))
        for original in group:
            value_to_center[original] = center

    return np.array([value_to_center[int(v)] for v in values], dtype=np.int32)


def _remove_near_duplicate_points(points, min_distance=4.0):
    """
    移除相邻的过近点，避免出现密集噪声顶点。
    """
    if len(points) < 2:
        return points

    filtered = [points[0]]
    for i in range(1, len(points)):
        prev = np.array(filtered[-1], dtype=np.float32)
        curr = np.array(points[i], dtype=np.float32)
        if np.linalg.norm(curr - prev) >= min_distance:
            filtered.append(points[i])

    if len(filtered) >= 2:
        first = np.array(filtered[0], dtype=np.float32)
        last = np.array(filtered[-1], dtype=np.float32)
        if np.linalg.norm(first - last) < min_distance:
            filtered.pop()
    return np.array(filtered, dtype=np.int32)


def _remove_collinear_points(points, angle_tolerance_deg=10.0, min_distance=4.0):
    """
    循环删除冗余共线点和过近点。
    """
    if len(points) < 4:
        return points

    angle_tol = math.radians(angle_tolerance_deg)
    pts = [np.array(p, dtype=np.float32) for p in points]

    changed = True
    while changed and len(pts) > 3:
        changed = False
        keep = []
        n = len(pts)
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % n]

            prev_dist = np.linalg.norm(curr_pt - prev_pt)
            next_dist = np.linalg.norm(next_pt - curr_pt)
            if prev_dist < min_distance or next_dist < min_distance:
                changed = True
                continue

            v1 = prev_pt - curr_pt
            v2 = next_pt - curr_pt
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                changed = True
                continue

            cos_theta = float(np.dot(v1, v2) / (norm1 * norm2))
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle = math.acos(cos_theta)

            # 接近 180 度视为共线点，删除中间点
            if abs(math.pi - angle) <= angle_tol:
                changed = True
                continue

            keep.append(curr_pt)

        if len(keep) >= 3:
            pts = keep
        else:
            break

    return np.array([[int(round(p[0])), int(round(p[1]))] for p in pts], dtype=np.int32)


def simplify_room_contour(
    points,
    manhattan_tolerance=4,
    min_distance=5.0,
    collinear_angle_tolerance_deg=10.0,
    dp_epsilon_ratio=0.004,
):
    """
    对房间轮廓进行简化：
    1) 曼哈顿化对齐（x/y 近值对齐）
    2) 近点去重
    3) 共线点删除
    4) Douglas-Peucker 二次简化
    """
    if points is None or len(points) < 3:
        return points

    pts = np.array(points, dtype=np.int32).reshape(-1, 2)

    # 1) 曼哈顿化
    pts[:, 0] = _snap_axis(pts[:, 0], manhattan_tolerance)
    pts[:, 1] = _snap_axis(pts[:, 1], manhattan_tolerance)

    # 2) 近点去重
    pts = _remove_near_duplicate_points(pts, min_distance=min_distance)
    if len(pts) < 3:
        return pts

    # 3) 共线点删除
    pts = _remove_collinear_points(
        pts,
        angle_tolerance_deg=collinear_angle_tolerance_deg,
        min_distance=min_distance,
    )
    if len(pts) < 3:
        return pts

    # 4) Douglas-Peucker 二次简化
    contour = pts.reshape(-1, 1, 2).astype(np.int32)
    epsilon = dp_epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)

    # 兜底：防止过度简化导致退化
    if len(approx) < 3:
        return pts
    return approx.astype(np.int32)


def exclude_door_from_room(room_approx, doors_and_windows, image_path, show_result=False):
    """
    从房间多边形近似中排除门区域内的点
    :param room_approx: 房间轮廓的多边形近似点
    :param doors_and_windows: 门窗检测结果字典
    :param image_path: 图像路径
    :param show_result: 是否显示结果图像
    :return: 排除门区域后的点数组
    """
    doors = doors_and_windows["doors"]
    room_approx = room_approx.reshape(-1, 2)  # 确保approx是二维数组
    result = []
    removed_count = 0
    protected_in_door_count = 0
    
    for point in room_approx:
        is_in_door = False
        for door in doors:
            x, y, w, h = door
            if (x <= point[0] <= x + w) and (y <= point[1] <= y + h):
                is_in_door = True
                break
        if not is_in_door:
            result.append(point)

    # 只有在需要显示时才绘制结果
    if show_result and len(result) > 0:
        img = cv2.imread(image_path)
        if img is not None:
            box = np.int64(result)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)  # 红色轮廓

            # 用红色圆点标记并写出坐标
            for point in result:
                x, y = point
                cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(img, f"({x},{y})", (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Exclude Door from Room")
            plt.axis('off')
            plt.show()

    return np.array(result)


def process_all_rooms(
    approx_points,
    doors_and_windows,
    image_path,
    manhattan_tolerance=4,
    min_distance=5.0,
    collinear_angle_tolerance_deg=10.0,
    dp_epsilon_ratio=0.004,
):
    """
    处理所有房间的轮廓点，排除门区域内的点
    :param approx_points: 所有房间的多边形近似点字典
    :param doors_and_windows: 门窗检测结果
    :param image_path: 图像路径
    :return: 处理后的房间轮廓点字典
    """
    processed_rooms = {}
    
    # 批量处理所有房间，不显示单个结果
    for room_name, approx_list in approx_points.items():
        processed_rooms[room_name] = []
        for room_approx in approx_list:
            refined_points = exclude_door_from_room(
                room_approx,
                doors_and_windows,
                image_path,
                show_result=False,
            )
            if len(refined_points) > 0:
                simplified_points = simplify_room_contour(
                    refined_points,
                    manhattan_tolerance=manhattan_tolerance,
                    min_distance=min_distance,
                    collinear_angle_tolerance_deg=collinear_angle_tolerance_deg,
                    dp_epsilon_ratio=dp_epsilon_ratio,
                )
                if len(simplified_points) >= 3:
                    processed_rooms[room_name].append(simplified_points)
                    print(
                        f"房间 {room_name}: 原始点数 {len(room_approx)}, "
                        f"去门后点数 {len(refined_points)}, 简化后点数 {len(simplified_points)}"
                    )
                else:
                    # 简化退化时，回退到去门后的点
                    processed_rooms[room_name].append(refined_points)
                    print(
                        f"房间 {room_name}: 原始点数 {len(room_approx)}, "
                        f"去门后点数 {len(refined_points)}, 简化退化，回退原结果"
                    )
    
    # 处理完所有房间后，统一显示结果
    display_all_processed_rooms(processed_rooms, image_path)
    
    return processed_rooms


def display_all_processed_rooms(processed_rooms, image_path):
    """
    显示所有处理后房间的轮廓点
    :param processed_rooms: 处理后的房间轮廓点字典
    :param image_path: 图像路径
    """
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像文件，跳过显示")
        return
    
    # 为不同房间使用不同颜色
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 黄色
        (128, 0, 128),  # 紫色
        (255, 165, 0),  # 橙色
    ]
    
    color_idx = 0
    for room_name, room_contours in processed_rooms.items():
        if len(room_contours) == 0:
            continue
            
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        for refined_points in room_contours:
            if len(refined_points) > 2:  # 至少需要3个点才能绘制轮廓
                # 绘制轮廓
                box = np.int64(refined_points)
                cv2.drawContours(img, [box], 0, color, 3)
                
                # 绘制关键转折点
                for point in refined_points:
                    x, y = point
                    cv2.circle(img, (x, y), 8, color, -1)
                    cv2.putText(img, f"({x},{y})", (x+5, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                # 在轮廓中心附近添加房间名称标签
                # if len(refined_points) > 0:
                #     center_x = int(np.mean([p[0] for p in refined_points]))
                #     center_y = int(np.mean([p[1] for p in refined_points]))
                #     cv2.putText(img, room_name, (center_x, center_y), 
                #               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    
    # 保存图像到当前运行输出目录
    import os
    output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'step4_room_contours_processed.png')
    cv2.imwrite(output_path, img)
    print(f"房间轮廓处理结果已保存到: {output_path}")
    
    # # 显示最终结果
    # plt.figure(figsize=(15, 12))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title("所有房间轮廓处理结果（排除门区域后）")
    # plt.axis('off')
    # plt.show()
    
    print(f"已显示 {len(processed_rooms)} 个房间的处理结果")
