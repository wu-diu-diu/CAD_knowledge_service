"""
门窗检测模块
负责检测和分类青色区域中的门、窗和其他物体
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


def _find_room_for_center(center, room_contours_by_name, nearest_threshold=30.0):
    """
    判断门中心点属于哪个房间轮廓。
    返回: (room_name, mode, distance)
    mode: inside / nearest / unknown
    """
    if not room_contours_by_name:
        return None, "unknown", None

    cx, cy = float(center[0]), float(center[1])
    nearest_room = None
    nearest_abs_dist = None

    for room_name, contour_list in room_contours_by_name.items():
        for contour in contour_list:
            contour_arr = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            if len(contour_arr) < 3:
                continue
            signed_dist = float(cv2.pointPolygonTest(contour_arr, (cx, cy), True))
            if signed_dist >= 0:
                return room_name, "inside", signed_dist
            abs_dist = abs(signed_dist)
            if nearest_abs_dist is None or abs_dist < nearest_abs_dist:
                nearest_abs_dist = abs_dist
                nearest_room = room_name

    if nearest_room is not None and nearest_abs_dist is not None and nearest_abs_dist <= float(nearest_threshold):
        return nearest_room, "nearest", -nearest_abs_dist
    return None, "unknown", None


def _normalize_candidate_points(room_door_candidates):
    """
    统一room_door_candidates格式为:
    {room_name: [(x,y), ...]}
    """
    normalized = {}
    if not room_door_candidates:
        return normalized
    for room_name, items in room_door_candidates.items():
        points = []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and "point" in item:
                p = item["point"]
            else:
                p = item
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            points.append((float(p[0]), float(p[1])))
        if points:
            normalized[room_name] = points
    return normalized


def _cluster_points(points, eps=22.0):
    """
    简单距离聚类（无依赖），返回多个簇，每簇为点列表。
    """
    if not points:
        return []
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    used = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if used[i]:
            continue
        queue = [i]
        used[i] = True
        comp = [i]
        while queue:
            cur = queue.pop(0)
            d = np.linalg.norm(pts - pts[cur], axis=1)
            neighbors = np.where((d <= eps) & (~used))[0].tolist()
            for nb in neighbors:
                used[nb] = True
                queue.append(nb)
                comp.append(nb)
        clusters.append(pts[comp])
    return clusters


def _build_room_candidate_clusters(room_door_candidates, eps=22.0):
    """
    构建每个房间的门候选簇:
    {
      "room_name": str,
      "cluster_id": str,
      "center": [x,y],
      "bbox": [minx,miny,maxx,maxy],
      "point_count": int
    }
    """
    points_by_room = _normalize_candidate_points(room_door_candidates)
    clusters = []
    for room_name, points in points_by_room.items():
        group_list = _cluster_points(points, eps=eps)
        for idx, grp in enumerate(group_list):
            xs = grp[:, 0]
            ys = grp[:, 1]
            cluster = {
                "room_name": room_name,
                "cluster_id": f"{room_name}#{idx+1}",
                "center": [float(np.mean(xs)), float(np.mean(ys))],
                "bbox": [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))],
                "point_count": int(len(grp)),
            }
            clusters.append(cluster)
    return clusters


def _point_to_bbox_distance(px, py, bbox_xyxy):
    minx, miny, maxx, maxy = bbox_xyxy
    dx = 0.0 if minx <= px <= maxx else min(abs(px - minx), abs(px - maxx))
    dy = 0.0 if miny <= py <= maxy else min(abs(py - miny), abs(py - maxy))
    return float(np.hypot(dx, dy))


def _match_door_to_room_by_candidates(door_bbox, candidate_clusters):
    """
    最小可用版匹配：
    - 按 door_center 到 cluster_center 最近
    - 用动态阈值过滤（过远则不匹配）
    """
    if not candidate_clusters:
        return None

    x, y, w, h = door_bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    door_diag = float(np.hypot(w, h))
    max_center_dist = max(24.0, 1.2 * door_diag)

    best = None
    for cluster in candidate_clusters:
        ccx, ccy = cluster["center"]
        d_center = float(np.hypot(cx - ccx, cy - ccy))
        d_bbox = _point_to_bbox_distance(cx, cy, cluster["bbox"])
        # 简单加权，中心距离主导
        score = d_center + 0.35 * d_bbox
        if best is None or score < best["score"]:
            best = {
                "cluster": cluster,
                "d_center": d_center,
                "d_bbox": d_bbox,
                "score": score,
            }

    if best is None or best["d_center"] > max_center_dist:
        return None

    # 归一化置信度（仅用于展示）
    conf = max(0.0, min(1.0, 1.0 - best["d_center"] / max_center_dist))
    return {
        "room_name": best["cluster"]["room_name"],
        "cluster_id": best["cluster"]["cluster_id"],
        "cluster_center": best["cluster"]["center"],
        "point_count": best["cluster"]["point_count"],
        "distance_center": best["d_center"],
        "distance_bbox": best["d_bbox"],
        "score": best["score"],
        "confidence": conf,
    }


def _rank_door_room_candidates(door_bbox, candidate_clusters):
    """
    为单个门输出按分数排序的房间候选（每个房间仅保留最佳簇）。
    返回: [match_dict, ...]，按score升序
    """
    if not candidate_clusters:
        return []

    x, y, w, h = door_bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    door_diag = float(np.hypot(w, h))
    max_center_dist = max(24.0, 1.2 * door_diag)

    best_by_room = {}
    for cluster in candidate_clusters:
        ccx, ccy = cluster["center"]
        d_center = float(np.hypot(cx - ccx, cy - ccy))
        d_bbox = _point_to_bbox_distance(cx, cy, cluster["bbox"])
        score = d_center + 0.35 * d_bbox
        item = {
            "room_name": cluster["room_name"],
            "cluster_id": cluster["cluster_id"],
            "cluster_center": cluster["center"],
            "point_count": cluster["point_count"],
            "distance_center": d_center,
            "distance_bbox": d_bbox,
            "score": score,
            "confidence": max(0.0, min(1.0, 1.0 - d_center / max_center_dist)),
            "is_valid": bool(d_center <= max_center_dist),
        }
        prev = best_by_room.get(cluster["room_name"])
        if prev is None or item["score"] < prev["score"]:
            best_by_room[cluster["room_name"]] = item

    ranked = sorted(best_by_room.values(), key=lambda it: it["score"])
    return ranked


def _load_step3_font(font_size, font_path=None):
    if ImageFont is None:
        return None
    candidates = [font_path] if font_path else []
    candidates.extend(
        [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, int(font_size))
            except Exception:
                continue
    return None


def _draw_step3_labels(img_bgr, text_items, font_size=22, font_path=None):
    """
    用PIL绘制标签，保证中文可显示。
    text_items: List[(text, (x,y), (b,g,r))]
    """
    if not text_items:
        return img_bgr

    font = _load_step3_font(font_size, font_path=font_path)
    if Image is None or ImageDraw is None or font is None:
        # 回退: OpenCV，中文可能显示为问号
        for text, pos, color_bgr in text_items:
            cv2.putText(
                img_bgr,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_bgr,
                2,
                cv2.LINE_AA,
            )
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    drawer = ImageDraw.Draw(pil_img)
    for text, pos, color_bgr in text_items:
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        drawer.text((int(pos[0]), int(pos[1])), text, fill=color_rgb, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_quarter_circles(roi, contour):
    """
    检测轮廓中是否包含四分之一圆形状（门的特征）
    :param roi: 感兴趣区域
    :param contour: 轮廓
    :return: 检测到的四分之一圆数量 (0, 1, 2)
    """
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.medianBlur(gray_roi, 5)
    
    h, w = gray_roi.shape
    quarter_circles_count = 0
    
    # 1. 使用Hough圆检测，调整参数以更好地检测四分之一圆
    circles = cv2.HoughCircles(
        gray_roi,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(w, h)//4,
        param1=50,
        param2=25,  # 降低阈值以检测更多圆弧
        minRadius=min(w, h)//8,
        maxRadius=min(w, h)//2
    )
    
    if circles is not None:
        quarter_circles_count = len(circles[0])
    
    # 2. 基于轮廓形状分析检测四分之一圆
    try:
        if len(contour) < 5:
            return quarter_circles_count
            
        # 简化轮廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx_contour) < 4:
            return quarter_circles_count
            
        # 计算轮廓的凸包缺陷
        hull = cv2.convexHull(approx_contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(approx_contour, hull)
            if defects is not None:
                # 统计显著的凸包缺陷（可能对应四分之一圆）
                significant_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = d / 256.0
                    # 如果缺陷深度足够大，认为是四分之一圆弧
                    if depth > min(w, h) * 0.05:
                        significant_defects += 1
                
                # 根据缺陷数量推测四分之一圆数量
                if significant_defects >= 2:
                    quarter_circles_count = max(quarter_circles_count, 2)
                elif significant_defects >= 1:
                    quarter_circles_count = max(quarter_circles_count, 1)
                    
    except cv2.error as e:
        print(f"轮廓分析失败，跳过: {e}")
        pass
    
    # 3. 基于面积比值和形状特征
    contour_area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    
    if bbox_area > 0:
        area_ratio = contour_area / bbox_area
        aspect_ratio = w / h if h > 0 else 0
        
        # 单扇门：接近正方形，面积比约0.6-0.8
        if 0.8 <= aspect_ratio <= 1.2 and 0.6 <= area_ratio <= 0.8:
            quarter_circles_count = max(quarter_circles_count, 1)
        # 双扇门：长方形，面积比约0.5-0.7
        elif 1.5 <= aspect_ratio <= 2.5 and 0.5 <= area_ratio <= 0.7:
            quarter_circles_count = max(quarter_circles_count, 2)
    
    return min(quarter_circles_count, 2)  # 最多返回2个


def is_door(roi, contour):
    """
    判断是否为门（包含1个或2个四分之一圆）
    :param roi: 感兴趣区域
    :param contour: 轮廓
    :return: 是否为门
    """
    quarter_circles = detect_quarter_circles(roi, contour)
    return quarter_circles > 0


def find_door_and_window(
    image_path,
    room_contours_by_name=None,
    room_door_candidates=None,
    assign_nearest_threshold=30.0,
    label_font_size=22,
    font_path=None,
):
    """
    检测图像中的门窗和其他青色物体
    :param image_path: 输入图片路径
    :return: 包含doors、windows、others列表的字典
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("图像读取失败，请检查路径。")

    # 1. 提取青色区域（门窗常用色）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 青色范围（可根据实际调整）
    lower_cyan = np.array([80, 50, 50])
    upper_cyan = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

    # 2. 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    assign_nearest_threshold = float(assign_nearest_threshold)
    label_font_size = int(label_font_size)
    candidate_clusters = _build_room_candidate_clusters(room_door_candidates, eps=22.0)
    print(f"step3 候选簇数量: {len(candidate_clusters)}")

    # 4. 按层次结构分类：窗 → 门 → 其他
    doors, windows, others = [], [], []
    door_assignments = []
    step3_text_items = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(cnt)
        
        # 跳过面积过小的轮廓
        if area < 100:
            continue
            
        roi = img[y:y+h, x:x+w]
        
        # 第一优先级：判断是否为窗（细长矩形）
        if aspect_ratio > 4 or aspect_ratio < 0.25:
            windows.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,255), 2)
            cv2.putText(img, 'Window', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            print(f"检测到窗：({x}, {y}), 尺寸: {w}x{h}, 长宽比: {aspect_ratio:.2f}")
            
        # 第二优先级：判断是否为门（包含四分之一圆）
        elif is_door(roi, cnt):
            doors.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(img, 'Door', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cx, cy = x + w / 2.0, y + h / 2.0
            ranked_matches = _rank_door_room_candidates((x, y, w, h), candidate_clusters)
            primary_match = ranked_matches[0] if ranked_matches else None
            secondary_match = ranked_matches[1] if len(ranked_matches) >= 2 else None
            if primary_match is not None:
                room_name = primary_match["room_name"]
                assign_mode = "candidate_ranked"
                assign_dist = primary_match["distance_center"]
                ccx, ccy = primary_match["cluster_center"]
                cv2.line(img, (int(round(cx)), int(round(cy))), (int(round(ccx)), int(round(ccy))), (255, 0, 0), 2)
                if secondary_match is not None:
                    sccx, sccy = secondary_match["cluster_center"]
                    cv2.line(img, (int(round(cx)), int(round(cy))), (int(round(sccx)), int(round(sccy))), (255, 255, 0), 2)
            else:
                room_name, assign_mode, assign_dist = _find_room_for_center(
                    (cx, cy),
                    room_contours_by_name,
                    nearest_threshold=assign_nearest_threshold,
                )
                primary_match = None
                secondary_match = None
            cv2.circle(img, (int(round(cx)), int(round(cy))), 4, (255, 0, 255), -1)
            secondary_room = secondary_match["room_name"] if secondary_match is not None else None
            label_text = f"主:{room_name if room_name else '未知'} 副:{secondary_room if secondary_room else '未知'}"
            if assign_mode == "candidate_ranked" and room_name:
                label_text = f"{label_text}(候选)"
            elif assign_mode == "nearest" and room_name:
                label_text = f"房间:{room_name}(邻近)"
            step3_text_items.append((label_text, (x, y + h + 6), (255, 0, 255)))
            door_assignments.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "center": [float(cx), float(cy)],
                    "room": room_name,
                    "primary_room": room_name,
                    "secondary_room": secondary_room,
                    "assign_mode": assign_mode,
                    "distance": float(assign_dist) if assign_dist is not None else None,
                    "match": {
                        "primary": primary_match,
                        "secondary": secondary_match,
                    },
                }
            )
                
        # 第三优先级：其余暂时标记为门
        else:
            doors.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            ## 表示没有识别成功但是仍标记为门
            cv2.putText(img, 'Door(o)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cx, cy = x + w / 2.0, y + h / 2.0
            ranked_matches = _rank_door_room_candidates((x, y, w, h), candidate_clusters)
            primary_match = ranked_matches[0] if ranked_matches else None
            secondary_match = ranked_matches[1] if len(ranked_matches) >= 2 else None
            if primary_match is not None:
                room_name = primary_match["room_name"]
                assign_mode = "candidate_ranked"
                assign_dist = primary_match["distance_center"]
                ccx, ccy = primary_match["cluster_center"]
                cv2.line(img, (int(round(cx)), int(round(cy))), (int(round(ccx)), int(round(ccy))), (255, 0, 0), 2)
                if secondary_match is not None:
                    sccx, sccy = secondary_match["cluster_center"]
                    cv2.line(img, (int(round(cx)), int(round(cy))), (int(round(sccx)), int(round(sccy))), (255, 255, 0), 2)
            else:
                room_name, assign_mode, assign_dist = _find_room_for_center(
                    (cx, cy),
                    room_contours_by_name,
                    nearest_threshold=assign_nearest_threshold,
                )
                primary_match = None
                secondary_match = None
            cv2.circle(img, (int(round(cx)), int(round(cy))), 4, (255, 0, 255), -1)
            secondary_room = secondary_match["room_name"] if secondary_match is not None else None
            label_text = f"主:{room_name if room_name else '未知'} 副:{secondary_room if secondary_room else '未知'}"
            if assign_mode == "candidate_ranked" and room_name:
                label_text = f"{label_text}(候选)"
            elif assign_mode == "nearest" and room_name:
                label_text = f"房间:{room_name}(邻近)"
            step3_text_items.append((label_text, (x, y + h + 6), (255, 0, 255)))
            door_assignments.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "center": [float(cx), float(cy)],
                    "room": room_name,
                    "primary_room": room_name,
                    "secondary_room": secondary_room,
                    "assign_mode": assign_mode,
                    "distance": float(assign_dist) if assign_dist is not None else None,
                    "match": {
                        "primary": primary_match,
                        "secondary": secondary_match,
                    },
                }
            )

    # 最后统一绘制中文标签，避免OpenCV中文乱码
    img = _draw_step3_labels(
        img,
        step3_text_items,
        font_size=label_font_size,
        font_path=font_path,
    )

    # 保存图像到当前运行输出目录
    output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'step3_door_window_detection.png')
    cv2.imwrite(output_path, img)
    print(f"门窗检测结果已保存到: {output_path}")

    # plt.figure(figsize=(12, 10))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title("Door and Window Detection")
    # plt.axis('off')
    # plt.show()

    assigned_count = len([d for d in door_assignments if d.get("room")])
    print(f"门中心归属结果: 总门数={len(door_assignments)}, 已归属={assigned_count}, 未归属={len(door_assignments)-assigned_count}")
    return {"doors": doors, "windows": windows, "others": [], "door_assignments": door_assignments}
