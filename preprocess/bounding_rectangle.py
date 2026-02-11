"""
最小外接矩形计算模块
计算每个房间所有轮廓点的最小外接矩形
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_min_bounding_rectangle(points):
    """
    计算点集的最小外接矩形
    :param points: 点坐标列表 [[x1, y1], [x2, y2], ...]
    :return: 最小外接矩形的四个顶点坐标
    """
    if len(points) < 3:
        # 如果点数少于3个，无法形成矩形，返回普通边界框
        points_array = np.array(points)
        x_min, y_min = np.min(points_array, axis=0)
        x_max, y_max = np.max(points_array, axis=0)
        return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    
    # 转换为OpenCV需要的格式
    points_array = np.array(points, dtype=np.float32)
    
    # 计算最小外接矩形
    rect = cv2.minAreaRect(points_array)
    
    # 获取矩形的四个顶点
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    # 按顺序排列顶点：左下、右下、右上、左上
    center = np.mean(box, axis=0)
    
    # 计算每个点相对于中心的角度
    angles = []
    for point in box:
        angle = np.arctan2(point[1] - center[1], point[0] - center[0])
        angles.append(angle)
    
    # 按角度排序
    sorted_indices = np.argsort(angles)
    sorted_box = box[sorted_indices]
    
    return sorted_box.tolist()


def _build_reference_mask(contour_list, padding=4):
    """
    将房间轮廓转为局部二值掩膜，用于IoU和覆盖率计算。
    """
    all_points = []
    for contour in contour_list:
        contour_arr = np.array(contour, dtype=np.int32).reshape(-1, 2)
        if len(contour_arr) >= 3:
            all_points.append(contour_arr)

    if not all_points:
        return None, None, None

    merged = np.vstack(all_points)
    min_x = int(np.min(merged[:, 0])) - padding
    min_y = int(np.min(merged[:, 1])) - padding
    max_x = int(np.max(merged[:, 0])) + padding
    max_y = int(np.max(merged[:, 1])) + padding

    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)
    mask = np.zeros((height, width), dtype=np.uint8)

    for contour in all_points:
        shifted = contour.copy()
        shifted[:, 0] -= min_x
        shifted[:, 1] -= min_y
        cv2.fillPoly(mask, [shifted], 1)

    return mask, min_x, min_y


def _polygon_to_mask(polygon, shape, offset_x, offset_y):
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    if len(points) < 3:
        return mask
    points[:, 0] -= offset_x
    points[:, 1] -= offset_y
    cv2.fillPoly(mask, [points], 1)
    return mask


def _rectangles_to_mask(rectangles, shape, offset_x, offset_y):
    mask = np.zeros(shape, dtype=np.uint8)
    for rect in rectangles:
        points = np.array(rect, dtype=np.int32).reshape(-1, 2)
        if len(points) < 3:
            continue
        shifted = points.copy()
        shifted[:, 0] -= offset_x
        shifted[:, 1] -= offset_y
        cv2.fillPoly(mask, [shifted], 1)
    return mask


def _polygons_to_mask(polygons, shape, offset_x, offset_y):
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        if len(points) < 3:
            continue
        shifted = points.copy()
        shifted[:, 0] -= offset_x
        shifted[:, 1] -= offset_y
        cv2.fillPoly(mask, [shifted], 1)
    return mask


def _calc_metrics(reference_mask, candidate_mask):
    ref = reference_mask.astype(bool)
    cand = candidate_mask.astype(bool)
    inter = np.logical_and(ref, cand).sum()
    union = np.logical_or(ref, cand).sum()
    cand_area = cand.sum()
    ref_area = ref.sum()

    iou = float(inter / union) if union > 0 else 0.0
    over_coverage = float(np.logical_and(cand, ~ref).sum() / cand_area) if cand_area > 0 else 1.0
    coverage = float(inter / ref_area) if ref_area > 0 else 0.0
    return iou, over_coverage, coverage


def _largest_rectangle_in_histogram(heights):
    """
    在直方图中查找最大矩形，返回(area, left, right, height)。
    """
    max_area = 0
    best_left = best_right = best_height = 0
    stack = []

    for i in range(len(heights) + 1):
        current_height = heights[i] if i < len(heights) else 0
        start = i
        while stack and stack[-1][1] > current_height:
            idx, h = stack.pop()
            area = h * (i - idx)
            if area > max_area:
                max_area = area
                best_left = idx
                best_right = i - 1
                best_height = h
            start = idx
        if not stack or stack[-1][1] < current_height:
            stack.append((start, current_height))

    return max_area, best_left, best_right, best_height


def _largest_inner_rectangle(binary_mask):
    """
    在二值掩膜中找最大轴对齐内接矩形，返回 (x, y, w, h, area)。
    """
    height, width = binary_mask.shape
    heights = [0] * width
    best = None

    for y in range(height):
        row = binary_mask[y]
        for x in range(width):
            heights[x] = heights[x] + 1 if row[x] > 0 else 0
        area, left, right, h = _largest_rectangle_in_histogram(heights)
        if area <= 0:
            continue
        top = y - h + 1
        w = right - left + 1
        candidate = (left, top, w, h, area)
        if best is None or area > best[4]:
            best = candidate

    return best


def _decompose_to_multi_rectangles(reference_mask, min_rect_area=120, max_rectangles=8, target_coverage=0.97):
    """
    使用“最大内接矩形迭代”将不规则区域分解为多个矩形。
    """
    remaining = reference_mask.copy().astype(np.uint8)
    rectangles = []
    ref_area = int(reference_mask.sum())
    if ref_area <= 0:
        return rectangles

    covered = np.zeros_like(reference_mask, dtype=np.uint8)
    while len(rectangles) < max_rectangles:
        best = _largest_inner_rectangle(remaining)
        if best is None:
            break
        x, y, w, h, area = best
        if area < min_rect_area:
            break

        rect = [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]
        rectangles.append(rect)
        remaining[y:y + h, x:x + w] = 0
        covered[y:y + h, x:x + w] = 1

        coverage = float(np.logical_and(covered > 0, reference_mask > 0).sum() / ref_area)
        if coverage >= target_coverage:
            break

    return rectangles


def _globalize_rectangles(local_rectangles, offset_x, offset_y):
    global_rectangles = []
    for rect in local_rectangles:
        global_rectangles.append([[p[0] + offset_x, p[1] + offset_y] for p in rect])
    return global_rectangles


def _dedupe_points(points):
    deduped = []
    for p in points:
        px, py = int(p[0]), int(p[1])
        if not deduped or deduped[-1] != [px, py]:
            deduped.append([px, py])
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p2[1]) == (p2[1] - p1[1]) * (p3[0] - p2[0])


def _build_snap_map(values, tolerance=1):
    if not values:
        return {}
    unique_vals = sorted({int(v) for v in values})
    clusters = []
    current = [unique_vals[0]]
    for v in unique_vals[1:]:
        if v - current[-1] <= tolerance:
            current.append(v)
        else:
            clusters.append(current)
            current = [v]
    clusters.append(current)

    value_map = {}
    for cluster in clusters:
        snapped = int(round(sum(cluster) / len(cluster)))
        for v in cluster:
            value_map[v] = snapped
    return value_map


def _snap_axis_points(points, tolerance=1):
    if not points:
        return []
    x_map = _build_snap_map([p[0] for p in points], tolerance=tolerance)
    y_map = _build_snap_map([p[1] for p in points], tolerance=tolerance)
    return [[x_map[int(p[0])], y_map[int(p[1])]] for p in points]


def _simplify_polygon_points(points):
    pts = _dedupe_points(points)
    if len(pts) <= 3:
        return pts

    # 将1像素级抖动坐标吸附到统一轴上，减少无意义拐点
    pts = _snap_axis_points(pts, tolerance=1)
    pts = _dedupe_points(pts)
    if len(pts) <= 3:
        return pts

    changed = True
    while changed and len(pts) > 3:
        changed = False
        new_pts = []
        total = len(pts)
        for i in range(total):
            prev_p = pts[(i - 1) % total]
            cur_p = pts[i]
            next_p = pts[(i + 1) % total]
            if _is_collinear(prev_p, cur_p, next_p):
                changed = True
                continue
            new_pts.append(cur_p)
        pts = _dedupe_points(new_pts)

    return pts


def _merge_rectangles_to_polygons(local_rectangles, mask_shape, offset_x, offset_y, min_area=80.0):
    """
    将多个矩形并集后提取外轮廓，并删除冗余共线点，得到更紧凑的多边形表示。
    """
    if not local_rectangles:
        return []

    union_mask = _rectangles_to_mask(local_rectangles, mask_shape, 0, 0)
    contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        contour_pts = contour.reshape(-1, 2).tolist()
        simplified_local = _simplify_polygon_points(contour_pts)
        if len(simplified_local) < 3:
            continue
        simplified_global = [[p[0] + offset_x, p[1] + offset_y] for p in simplified_local]
        polygons.append(simplified_global)

    return polygons


def calculate_room_bounding_rectangles(
    processed_rooms,
    regular_iou_threshold=0.90,
    regular_over_coverage_threshold=0.08,
    multi_min_rect_area=120,
    multi_max_rectangles=8,
    multi_target_coverage=0.97,
):
    """
    计算每个房间的最小外接矩形
    :param processed_rooms: 处理后的房间轮廓点字典
    :return: 房间名称到最小外接矩形顶点的字典
    """
    room_rectangles = {}
    
    for room_name, contour_list in processed_rooms.items():
        if not contour_list:
            continue
            
        # 收集该房间所有轮廓的所有点
        all_points = []
        for contour in contour_list:
            all_points.extend(contour)
        
        if len(all_points) > 0:
            # 候选A：单最小外接矩形
            single_rect = calculate_min_bounding_rectangle(all_points)

            reference_mask, offset_x, offset_y = _build_reference_mask(contour_list)
            if reference_mask is None:
                room_rectangles[room_name] = [single_rect]
                continue

            single_mask = _polygon_to_mask(single_rect, reference_mask.shape, offset_x, offset_y)
            single_iou, single_over, _ = _calc_metrics(reference_mask, single_mask)

            # 规则房间：直接采用单矩形
            if single_iou >= regular_iou_threshold and single_over <= regular_over_coverage_threshold:
                room_rectangles[room_name] = [single_rect]
                print(
                    f"房间 '{room_name}': 规则形状, 单矩形 "
                    f"(IoU={single_iou:.3f}, over={single_over:.3f})"
                )
                continue

            # 不规则房间：多矩形分解
            local_multi = _decompose_to_multi_rectangles(
                reference_mask,
                min_rect_area=multi_min_rect_area,
                max_rectangles=multi_max_rectangles,
                target_coverage=multi_target_coverage,
            )

            if not local_multi:
                room_rectangles[room_name] = [single_rect]
                print(
                    f"房间 '{room_name}': 分解失败，回退单矩形 "
                    f"(IoU={single_iou:.3f}, over={single_over:.3f})"
                )
                continue

            multi_rects = _globalize_rectangles(local_multi, offset_x, offset_y)
            multi_mask = _rectangles_to_mask(multi_rects, reference_mask.shape, offset_x, offset_y)
            multi_iou, multi_over, multi_cov = _calc_metrics(reference_mask, multi_mask)

            merged_polygons = _merge_rectangles_to_polygons(
                local_multi,
                reference_mask.shape,
                offset_x,
                offset_y,
            )
            merged_mask = _polygons_to_mask(merged_polygons, reference_mask.shape, offset_x, offset_y)
            merged_iou, merged_over, merged_cov = _calc_metrics(reference_mask, merged_mask)

            # 自适应选择：多矩形明显更好则采用，否则回退单矩形
            if (multi_iou > single_iou + 0.02) or (multi_over < single_over - 0.02):
                if merged_polygons:
                    room_rectangles[room_name] = merged_polygons
                    point_count = sum(len(poly) for poly in merged_polygons)
                    print(
                        f"房间 '{room_name}': 不规则形状, 采用并集轮廓({len(merged_polygons)}个) "
                        f"(points={point_count}, single_iou={single_iou:.3f}, "
                        f"merged_iou={merged_iou:.3f}, single_over={single_over:.3f}, "
                        f"merged_over={merged_over:.3f}, merged_cov={merged_cov:.3f})"
                    )
                else:
                    room_rectangles[room_name] = multi_rects
                    print(
                        f"房间 '{room_name}': 不规则形状, 采用多矩形({len(multi_rects)}个) "
                        f"(single_iou={single_iou:.3f}, multi_iou={multi_iou:.3f}, "
                        f"single_over={single_over:.3f}, multi_over={multi_over:.3f}, "
                        f"multi_cov={multi_cov:.3f})"
                    )
            else:
                room_rectangles[room_name] = [single_rect]
                print(
                    f"房间 '{room_name}': 多矩形提升不足，回退单矩形 "
                    f"(single_iou={single_iou:.3f}, multi_iou={multi_iou:.3f}, "
                    f"single_over={single_over:.3f}, multi_over={multi_over:.3f})"
                )
    
    return room_rectangles


def visualize_bounding_rectangles(room_rectangles, image_path):
    """
    可视化房间的最小外接矩形
    :param room_rectangles: 房间矩形字典
    :param image_path: 原始图像路径
    """
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像文件，跳过可视化")
        return
    
    # 为不同房间使用不同颜色
    colors = [
        (255, 0, 0),      # 蓝色
        (0, 255, 0),      # 绿色  
        (0, 0, 255),      # 红色
        (255, 255, 0),    # 青色
        (255, 0, 255),    # 紫色
        (0, 255, 255),    # 黄色
        (128, 0, 128),    # 深紫色
        (255, 165, 0),    # 橙色
        (0, 128, 128),    # 青绿色
        (128, 128, 0),    # 橄榄色
    ]
    
    color_idx = 0
    for room_name, room_shapes in room_rectangles.items():
        color = colors[color_idx % len(colors)]
        color_idx += 1

        for shape_idx, rectangle_points in enumerate(room_shapes):
            rect_points = np.array(rectangle_points, dtype=np.int32)
            cv2.drawContours(img, [rect_points], 0, color, 3)
            for i, point in enumerate(rect_points):
                x, y = point
                cv2.circle(img, (x, y), 6, color, -1)
                cv2.putText(
                    img,
                    f"{shape_idx + 1}-{i + 1}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        
        # # 在矩形中心添加房间名称
        # center_x = int(np.mean([p[0] for p in rect_points]))
        # center_y = int(np.mean([p[1] for p in rect_points]))
        # cv2.putText(img, room_name, (center_x-30, center_y), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        
        # # 计算和显示矩形面积
        # area = cv2.contourArea(rect_points)
        # cv2.putText(img, f"Area:{area:.0f}", (center_x-40, center_y+25), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    # 保存图像到当前运行输出目录
    output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'step5_room_bounding_rectangles.png')
    cv2.imwrite(output_path, img)
    print(f"最小外接矩形结果已保存到: {output_path}")
    
    # 显示结果
    # plt.figure(figsize=(15, 12))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title("房间最小外接矩形")
    # plt.axis('off')
    # plt.show()


def process_room_bounding_rectangles(processed_rooms, image_path):
    """
    处理房间最小外接矩形的完整流程
    :param processed_rooms: 处理后的房间轮廓点字典
    :param image_path: 图像路径
    :return: 房间矩形字典
    """
    print("=== 步骤5: 计算房间最小外接矩形 ===")
    
    # 计算最小外接矩形
    room_rectangles = calculate_room_bounding_rectangles(processed_rooms)
    
    # 显示统计信息
    print("\n最小外接表示计算完成:")
    for room_name, shapes in room_rectangles.items():
        total_area = 0.0
        for shape in shapes:
            shape_arr = np.array(shape)
            total_area += cv2.contourArea(np.array(shape, dtype=np.int32))
            width = np.max(shape_arr[:, 0]) - np.min(shape_arr[:, 0])
            height = np.max(shape_arr[:, 1]) - np.min(shape_arr[:, 1])
            print(f"  {room_name}: 子形状 宽{width:.1f} x 高{height:.1f}, 面积{cv2.contourArea(np.array(shape, dtype=np.int32)):.0f}")
        print(f"    {room_name}: 共 {len(shapes)} 个形状, 总面积{total_area:.0f}")
    
    # 可视化和保存
    visualize_bounding_rectangles(room_rectangles, image_path)
    
    print("=== 最小外接矩形处理完成 ===\n")
    
    return room_rectangles


def save_bounding_rectangles(room_rectangles, output_file):
    """
    保存房间外接表示(单矩形或多矩形)到JSON文件
    :param room_rectangles: 房间形状字典
    :param output_file: 输出文件路径
    """
    import json
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(room_rectangles, f, ensure_ascii=False, indent=2)
    
    print(f"最小外接矩形数据已保存到: {output_file}")


def demo_bounding_rectangles():
    """
    演示最小外接矩形功能
    """
    # 模拟processed_rooms数据
    sample_processed_rooms = {
        '客厅': [
            [[100, 100], [200, 120], [180, 200], [90, 180]],  # 不规则四边形
        ],
        '卧室': [
            [[300, 150], [450, 160], [440, 280], [310, 270]],  # 另一个不规则四边形
        ],
        '厨房': [
            [[500, 50], [600, 50], [600, 150], [500, 150]],   # 矩形
        ]
    }
    
    image_path = './images/all_room.png'
    
    try:
        # 处理最小外接矩形
        room_rectangles = process_room_bounding_rectangles(sample_processed_rooms, image_path)
        
        # 保存结果
        output_file = 'images/output/room_bounding_rectangles.json'
        save_bounding_rectangles(room_rectangles, output_file)
        
        return room_rectangles
        
    except Exception as e:
        print(f"演示失败: {e}")
        return None


if __name__ == "__main__":
    demo_bounding_rectangles()
