"""
门窗检测模块
负责检测和分类青色区域中的门、窗和其他物体
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def find_door_and_window(image_path):
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

    # 4. 按层次结构分类：窗 → 门 → 其他
    doors, windows, others = [], [], []
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
                
        # 第三优先级：其余暂时标记为门
        else:
            doors.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            ## 表示没有识别成功但是仍标记为门
            cv2.putText(img, 'Door(o)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # 保存图像到当前运行输出目录
    import os
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

    return {"doors": doors, "windows": windows, "others": []}
