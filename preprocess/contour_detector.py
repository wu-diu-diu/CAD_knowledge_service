"""
轮廓检测模块
负责检测图像中所有封闭空间的内轮廓，并匹配房间名称
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def _sanitize_filename(name):
    # 保留中文，替换文件系统非法字符
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, "_")
    return name.strip() or "room"


def _euclidean(p1, p2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))


def _dedupe_candidates(candidates, merge_dist=20.0):
    """
    对候选凹口点按距离去重，优先保留depth更大的点。
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: float(item["depth"]), reverse=True)
    kept = []
    for item in ordered:
        p = item["point"]
        if all(_euclidean(p, k["point"]) > merge_dist for k in kept):
            kept.append(item)
    return kept


def _extract_door_notch_candidates(contour):
    """
    基于凸包缺陷提取房间轮廓上的“门候选凹口”。
    返回: [{"point":[x,y], "depth":..., "span":...}, ...]
    """
    if contour is None or len(contour) < 20:
        return []

    contour = contour.astype(np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    arc_len = float(cv2.arcLength(contour, True))
    if arc_len <= 0:
        return []

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return []

    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return []
    if defects is None:
        return []

    min_depth = max(8.0, 0.02 * min(w, h))
    max_depth = max(min_depth * 2.0, 0.50 * max(w, h))
    min_span = max(12.0, 0.03 * arc_len)
    max_span = max(min_span * 2.0, 0.40 * arc_len)

    candidates = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = float(d / 256.0)
        if depth < min_depth or depth > max_depth:
            continue

        start = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]
        span = _euclidean(start, end)
        if span < min_span or span > max_span:
            continue

        # 门凹口一般既有一定“凹陷深度”，又有一定“开口宽度”
        depth_span_ratio = depth / max(span, 1e-6)
        if depth_span_ratio < 0.08:
            continue

        candidates.append(
            {
                "point": [int(far[0]), int(far[1])],
                "depth": float(depth),
                "span": float(span),
            }
        )

    merge_dist = max(14.0, 0.03 * max(w, h))
    return _dedupe_candidates(candidates, merge_dist=merge_dist)


def _fit_circle_kasa(points):
    """
    最小二乘拟合圆（Kasa）。
    返回: (center[x,y], radius, residual)
    """
    pts = np.array(points, dtype=np.float64)
    if pts.shape[0] < 6:
        return None, None, None
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack((x, y, np.ones_like(x)))
    b = -(x * x + y * y)
    try:
        coef, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    A, B, C = coef
    cx = -A / 2.0
    cy = -B / 2.0
    r_sq = cx * cx + cy * cy - C
    if r_sq <= 1e-6:
        return None, None, None
    radius = float(np.sqrt(r_sq))
    dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residual = float(np.mean(np.abs(dists - radius)))
    return np.array([cx, cy], dtype=np.float64), radius, residual


def _extract_door_arc_points(contour):
    """
    在房间轮廓上检测弧线点（门弧特征）。
    返回: [[x,y], ...]
    """
    if contour is None:
        return []
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 40:
        return []

    x, y, w, h = cv2.boundingRect(contour)
    diag = float(np.hypot(w, h))
    if diag < 10:
        return []

    win = max(18, min(48, int(0.08 * n)))
    step = max(2, win // 5)
    min_radius = max(6.0, 0.02 * diag)
    max_radius = max(min_radius * 2.0, 0.30 * diag)
    max_rel_residual = 0.22
    min_span_rad = 0.45
    max_span_rad = 2.55

    is_arc = np.zeros(n, dtype=bool)
    for start in range(0, n - win + 1, step):
        window = pts[start:start + win]
        center, radius, residual = _fit_circle_kasa(window)
        if center is None:
            continue
        if radius < min_radius or radius > max_radius:
            continue
        if residual / max(radius, 1e-6) > max_rel_residual:
            continue

        rel = window - center
        angles = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        span = float(abs(angles[-1] - angles[0]))
        if span < min_span_rad or span > max_span_rad:
            continue

        v1 = window[1:-1] - window[:-2]
        v2 = window[2:] - window[1:-1]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        nonzero = cross[np.abs(cross) > 1e-3]
        if len(nonzero) < max(6, win // 4):
            continue
        sign_ratio = max(float(np.mean(nonzero > 0)), float(np.mean(nonzero < 0)))
        if sign_ratio < 0.70:
            continue

        is_arc[start:start + win] = True

    idx = np.where(is_arc)[0]
    if len(idx) == 0:
        return []

    min_run = max(6, win // 4)
    runs = []
    run_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((run_start, prev))
            run_start, prev = i, i
    runs.append((run_start, prev))

    kept_idx = []
    for s, e in runs:
        if e - s + 1 >= min_run:
            kept_idx.extend(range(s, e + 1))
    if not kept_idx:
        return []

    kept = pts[np.array(kept_idx, dtype=np.int32)]
    return [[int(round(p[0])), int(round(p[1]))] for p in kept]


def _extract_arc_like_points_from_approx(approx):
    """
    兜底：从多边形近似点中提取“非正交连续段”作为圆弧候选。
    适用于CAD正交墙体场景：门弧通常表现为连续斜向线段。
    """
    if approx is None:
        return []
    pts = approx.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 4:
        return []

    def _is_axis_aligned(v):
        length = float(np.linalg.norm(v))
        if length < 1e-6:
            return True
        # 与x/y轴夹角很小时认为正交边
        axis_ratio = min(abs(v[0]), abs(v[1])) / length
        return axis_ratio < 0.12

    candidate_idx = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_cur = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = p_cur - p_prev
        v2 = p_next - p_cur
        if np.linalg.norm(v1) < 2.0 or np.linalg.norm(v2) < 2.0:
            continue
        if (not _is_axis_aligned(v1)) or (not _is_axis_aligned(v2)):
            candidate_idx.append(i)

    if not candidate_idx:
        return []

    # 保留连续索引段（环结构）
    idx_set = set(candidate_idx)
    runs = []
    visited = set()
    for i in candidate_idx:
        if i in visited:
            continue
        run = [i]
        visited.add(i)
        j = (i + 1) % n
        while j in idx_set and j not in visited:
            run.append(j)
            visited.add(j)
            j = (j + 1) % n
        runs.append(run)

    kept = []
    for run in runs:
        if len(run) >= 2:
            for idx in run:
                p = pts[idx]
                kept.append([int(round(p[0])), int(round(p[1]))])
    return kept


def find_all_inner_contours(image_path, room_dict, min_area=10000):
    """
    1.先找“候选内轮廓”，不是简单取最内层，而是用层级和面积筛选
    2.然后按每个房间中心点（OCR文本中心）去匹配“包含该点的最小面积轮廓”，作为该房间轮廓
    3.最后对这个轮廓做 approxPolyDP 多边形近似，得到“简化后的关键转折点”。
    :param image_path: 输入图片路径
    :param room_dict: 房间名称到中心坐标的字典
    :param min_area: 过滤小轮廓的最小面积
    :return: (inner_contours, approx_points, room_door_candidates)
             - 内轮廓列表、多边形近似点字典、房间门候选凹口点
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_RGB = cv2.imread(image_path)  # 默认读取为彩色 BGR
    if img is None:
        raise FileNotFoundError("图像读取失败，请检查路径。")

    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 3    # 35, 5
    )

    # 形态学闭运算，连接断线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 统一展开为“房间实例”，支持同名房间的多个中心点
    room_instances = []
    for room_name, centers in room_dict.items():
        if isinstance(centers, tuple) and len(centers) == 2:
            centers = [centers]
        if not isinstance(centers, list):
            continue
        for center_idx, center in enumerate(centers):
            if not isinstance(center, (tuple, list)) or len(center) != 2:
                continue
            room_instances.append((room_name, center_idx, int(center[0]), int(center[1])))

    # 查找所有轮廓
    result = img_RGB.copy()
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    inner_contours = []
    approx_points = defaultdict(list)
    room_door_candidates = defaultdict(list)
    room_visual_items = defaultdict(list)
    instance_best_idx = {}
    instance_best_area = {}
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, h in enumerate(hierarchy):
            # h[3] != -1 表示有父轮廓，判断面积足够大
            area = cv2.contourArea(contours[i])
            if h[3] != -1 and area > min_area:
                # 进一步判断该轮廓是封闭的,过滤掉面积过小的轮廓
                if cv2.isContourConvex(contours[i]) or cv2.arcLength(contours[i], True) > 0:
                    for room_name, center_idx, cx, cy in room_instances:
                        # pointPolygonTest 返回 >=0 表示点在轮廓内或在边上
                        if cv2.pointPolygonTest(contours[i], (cx, cy), False) >= 0:
                            instance_key = (room_name, center_idx)
                            best_area = instance_best_area.get(instance_key)
                            if best_area is None or area < best_area:
                                instance_best_area[instance_key] = area
                                instance_best_idx[instance_key] = i

        # 将每个实例匹配到的轮廓聚合到房间名，避免同一房间名重复追加相同轮廓
        room_matched_idx = defaultdict(set)
        for room_name, center_idx, cx, cy in room_instances:
            instance_key = (room_name, center_idx)
            contour_idx = instance_best_idx.get(instance_key)
            if contour_idx is None:
                continue
            if contour_idx in room_matched_idx[room_name]:
                continue

            room_matched_idx[room_name].add(contour_idx)
            contour = contours[contour_idx]
            inner_contours.append(contour)
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
            cv2.circle(result, (cx, cy), 8, (0, 0, 255), -1)

            # 多边形近似获取关键转折点，并添加到结果列表
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_points[room_name].append(approx)
            # 用红色圆点标记关键转折点，并写出坐标
            for point in approx:
                x, y = point[0]
                cv2.circle(result, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(result, f"({x},{y})", (x+5, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

            # 提取门候选点：圆弧点（主）+ 凹口点（辅）+ approx兜底
            arc_points = _extract_door_arc_points(contour)
            notch_candidates = _extract_door_notch_candidates(contour)
            approx_arc_points = _extract_arc_like_points_from_approx(approx)
            candidate_points = set()
            for p in arc_points:
                candidate_points.add((int(p[0]), int(p[1])))
            for cand in notch_candidates:
                p = cand["point"]
                candidate_points.add((int(p[0]), int(p[1])))
            for p in approx_arc_points:
                candidate_points.add((int(p[0]), int(p[1])))

            for px, py in candidate_points:
                cv2.circle(result, (px, py), 5, (255, 0, 0), -1)  # 蓝色

            room_door_candidates[room_name].extend(
                [{"point": [int(px), int(py)]} for (px, py) in sorted(candidate_points)]
            )
            print(
                f"step2 候选点[{room_name}]: arc={len(arc_points)}, "
                f"notch={len(notch_candidates)}, approx_fallback={len(approx_arc_points)}, "
                f"merged={len(candidate_points)}"
            )
            room_visual_items[room_name].append(
                {
                    "contour": contour.copy(),
                    "approx": approx.copy(),
                    "center": (int(cx), int(cy)),
                    "door_candidate_points": [[int(px), int(py)] for (px, py) in sorted(candidate_points)],
                }
            )

        matched_count = sum(len(v) for v in room_matched_idx.values())
        print(f"共检测到 {matched_count} 个房间轮廓。")
        total_door_points = sum(len(v) for v in room_door_candidates.values())
        print(f"step2 门候选点数量: {total_door_points}")

        # 保存图像到当前运行输出目录
        import os
        output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
        os.makedirs(output_dir, exist_ok=True)
        output_dir_step2 = os.path.join(output_dir, "step2_contour_detection")
        os.makedirs(output_dir_step2, exist_ok=True)

        # 分房间可视化：从原图裁剪后单独保存
        saved_count = 0
        for room_name, items in room_visual_items.items():
            safe_name = _sanitize_filename(room_name)
            for idx, item in enumerate(items):
                contour = item["contour"]
                approx = item["approx"]
                cx, cy = item["center"]
                door_candidate_points = item["door_candidate_points"]

                x, y, w, h = cv2.boundingRect(contour)
                margin = 36
                x0 = max(0, x - margin)
                y0 = max(0, y - margin)
                x1 = min(img_RGB.shape[1], x + w + margin)
                y1 = min(img_RGB.shape[0], y + h + margin)
                crop = img_RGB[y0:y1, x0:x1].copy()

                # 绘制房间轮廓（绿色）
                shifted_contour = contour.copy()
                shifted_contour[:, 0, 0] -= x0
                shifted_contour[:, 0, 1] -= y0
                cv2.drawContours(crop, [shifted_contour], -1, (0, 255, 0), 3)

                # 绘制OCR中心（红色）
                cv2.circle(crop, (cx - x0, cy - y0), 8, (0, 0, 255), -1)

                # 绘制关键转折点（红色）
                for point in approx:
                    px, py = point[0]
                    cv2.circle(crop, (int(px - x0), int(py - y0)), 7, (0, 0, 255), -1)

                # 绘制门候选点（蓝色）
                for px, py in door_candidate_points:
                    cv2.circle(crop, (px - x0, py - y0), 7, (255, 0, 0), -1)

                if len(items) == 1:
                    file_name = f"{safe_name}.png"
                else:
                    file_name = f"{safe_name}_{idx+1}.png"
                output_path = os.path.join(output_dir_step2, file_name)
                cv2.imwrite(output_path, crop)
                saved_count += 1

        print(f"step2分房间轮廓图已保存到: {output_dir_step2} (共 {saved_count} 张)")
        
        # plt.figure(figsize=(12, 10))
        # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # plt.title("All Inner Polygons")
        # plt.axis('off')
        # plt.show()

    return inner_contours, approx_points, dict(room_door_candidates)
