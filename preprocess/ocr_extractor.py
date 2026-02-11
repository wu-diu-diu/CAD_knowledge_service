"""
OCR文本提取模块
负责使用PaddleOCR提取房间名称和坐标
"""
import os
import pickle
from paddleocr import PaddleOCR
from collections import defaultdict


def extract_text_boxes(image_path, cache_dir='./cache'):
    """
    使用PaddleOCR提取图像中的文本框及其中心坐标。
    支持缓存机制以提高性能。
    :param image_path: 输入图片路径
    :param cache_dir: 缓存目录
    :return: 字典，键为唯一房间名，值为中心坐标元组 (x, y)
    """
    # cache_file = os.path.join(cache_dir, 'dic_cache.pkl')
    
    # # 检查缓存文件是否存在
    # if os.path.exists(cache_file):
    #     print("从缓存加载房间名称...")
    #     with open(cache_file, 'rb') as f:
    #         dic = pickle.load(f)
    # else:
    print("使用PaddleOCR提取房间名称...")
    ocr = PaddleOCR(use_doc_orientation_classify=False, 
                    use_doc_unwarping=False, 
                    use_textline_orientation=False)
    result = ocr.predict(image_path)
    
    # 先按原始房间名分组，保留同名房间的所有中心点
    grouped = defaultdict(list)
    for res in result:
        boxes = res.get('rec_boxes', [])
        texts = res.get('rec_texts', [])
        for box, text in zip(boxes, texts):
            if text.endswith('间') or text.endswith('室'):
                x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
                if center not in grouped[text]:
                    grouped[text].append(center)
        
        # # 保存到缓存
        # os.makedirs(cache_dir, exist_ok=True)
        # with open(cache_file, 'wb') as f:
        #     pickle.dump(dict(dic), f)
            
    # 再将同名房间拆分为唯一键:
    # 办公室 -> 办公室, 办公室1, 办公室2
    room_dict = {}
    for room_name, centers in grouped.items():
        ordered_centers = sorted(centers, key=lambda p: (p[1], p[0]))
        for idx, center in enumerate(ordered_centers):
            if idx == 0:
                key = room_name
            else:
                key = f"{room_name}{idx}"
            # 避免与OCR中本身存在的名称冲突（如“办公室1”）
            if key in room_dict:
                suffix = max(1, idx)
                while f"{room_name}{suffix}" in room_dict:
                    suffix += 1
                key = f"{room_name}{suffix}"
            room_dict[key] = center

    print(f"房间名称字典: {room_dict}")
    print(f"共识别出 {len(room_dict)} 个房间实例。")
    return room_dict
