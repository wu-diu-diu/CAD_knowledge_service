"""
轮廓检测模块
负责检测图像中所有封闭空间的内轮廓，并匹配房间名称
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def find_all_inner_contours(image_path, room_dict, min_area=10000):
    """
    1.先找“候选内轮廓”，不是简单取最内层，而是用层级和面积筛选
    2.然后按每个房间中心点（OCR文本中心）去匹配“包含该点的最小面积轮廓”，作为该房间轮廓
    3.最后对这个轮廓做 approxPolyDP 多边形近似，得到“简化后的关键转折点”。
    :param image_path: 输入图片路径
    :param room_dict: 房间名称到中心坐标的字典
    :param min_area: 过滤小轮廓的最小面积
    :return: (inner_contours, approx_points) - 内轮廓列表和多边形近似点字典
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
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inner_contours = []
    approx_points = defaultdict(list)
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

        matched_count = sum(len(v) for v in room_matched_idx.values())
        print(f"共检测到 {matched_count} 个房间轮廓。")

        # 保存图像到当前运行输出目录
        import os
        output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'step2_contour_detection.png')
        cv2.imwrite(output_path, result)
        print(f"轮廓检测结果已保存到: {output_path}")
        
        # plt.figure(figsize=(12, 10))
        # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # plt.title("All Inner Polygons")
        # plt.axis('off')
        # plt.show()

    return inner_contours, approx_points
