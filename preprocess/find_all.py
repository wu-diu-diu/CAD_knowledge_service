"""
CAD图纸房间分析主程序 - 重构版本
整合了OCR文本识别、轮廓检测、门窗识别和转折点处理的完整流程
"""
from .ocr_extractor import extract_text_boxes
from .contour_detector import find_all_inner_contours
from .door_window_detector import find_door_and_window
from .door_point_exclusion import process_all_rooms
from .bounding_rectangle import process_room_bounding_rectangles, save_bounding_rectangles
from .coordinate_converter import process_rooms_to_cad, DEFAULT_CAD_PARAMS
from .lighting_layout import process_room_lighting_layout
from .adaptive_shape_analyzer import process_adaptive_room_shapes, create_comparison_visualization
import os
import shutil
from datetime import datetime


def process_single_image(image_path, cad_params=None, save_to_file=True):
    """
    处理单个图像的完整流程
    :param image_path: 图像文件路径
    :param cad_params: CAD参数字典，如果为None则使用默认参数
    :param save_to_file: 是否保存到文件
    :return: 处理结果字典
    """
    print(f"=== 开始处理图像: {image_path} ===")

    # 为当前运行创建独立输出目录: images/output_YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("images", f"output_{timestamp}")
    if os.path.exists(output_dir):
        suffix = 1
        while os.path.exists(f"{output_dir}_{suffix}"):
            suffix += 1
        output_dir = f"{output_dir}_{suffix}"
    os.makedirs(output_dir, exist_ok=False)

    # 复制原始输入图像，方便与处理结果对照
    input_filename = os.path.basename(image_path)
    copied_input_path = os.path.join(output_dir, f"original_{input_filename}")
    shutil.copy2(image_path, copied_input_path)
    print(f"本次运行输出目录: {output_dir}")
    print(f"已复制原始图像到: {copied_input_path}")

    # 通过环境变量统一各步骤输出目录
    previous_output_dir = os.environ.get("CAD_STEP_OUTPUT_DIR")
    os.environ["CAD_STEP_OUTPUT_DIR"] = output_dir

    try:

        # 步骤1: 提取房间名称和坐标
        print("1. 提取房间名称和坐标...")
        room_dict = extract_text_boxes(image_path)

        # 步骤2: 检测房间轮廓
        print("2. 检测房间内轮廓...")
        inner_contours, approx_points, room_door_candidates = find_all_inner_contours(
            image_path,
            room_dict,
        )
        total_step2_door_candidates = sum(len(v) for v in room_door_candidates.values())
        print(f"step2 门候选凹口点: {total_step2_door_candidates}")

        # 步骤3: 检测门窗
        print("3. 检测门窗...")
        doors_and_windows = find_door_and_window(
            image_path,
            room_contours_by_name=approx_points,
            room_door_candidates=room_door_candidates,
        )
        print(f"检测结果: {len(doors_and_windows['doors'])} 个门, "
              f"{len(doors_and_windows['windows'])} 个窗")
        if "door_assignments" in doors_and_windows:
            assigned_count = len([d for d in doors_and_windows["door_assignments"] if d.get("room")])
            print(
                f"门中心归属: 总门数={len(doors_and_windows['door_assignments'])}, "
                f"已归属={assigned_count}, "
                f"未归属={len(doors_and_windows['door_assignments']) - assigned_count}"
            )

        # 步骤4: 处理房间轮廓点（排除门区域 + 轮廓简化）
        print("4. 处理房间轮廓点（排除门区域 + 轮廓简化）...")
        processed_rooms = process_all_rooms(
            approx_points,
            doors_and_windows,
            image_path,
            manhattan_tolerance=4,
            min_distance=5.0,
            collinear_angle_tolerance_deg=10.0,
            dp_epsilon_ratio=0.004,
        )

        # 步骤5: 计算房间最小外接矩形
        print("5. 计算房间最小外接矩形...")
        room_rectangles = process_room_bounding_rectangles(processed_rooms, image_path)

        # 步骤6：转换坐标为CAD坐标
        print("6. 转换坐标为CAD坐标...")
        cad_rooms = process_rooms_to_cad(room_rectangles, image_path, cad_params, save_to_file)

        # 步骤7: 房间网格离散化 + 灯具布置(测试链路)
        print("7. 房间网格离散化并生成灯具布置...")
        effective_cad_params = cad_params if cad_params is not None else DEFAULT_CAD_PARAMS
        lighting_payload = process_room_lighting_layout(
            room_rectangles=room_rectangles,
            image_path=image_path,
            cad_params=effective_cad_params,
            save_to_file=save_to_file,
        )

    # # 步骤7：自适应形状分析 (解决不规则房间形状问题)
    # print("7. 自适应形状分析...")
    # adaptive_shapes = process_adaptive_room_shapes(processed_rooms, image_path)

    # if save_to_file:
    #     # 创建形状方法对比可视化
    #     print("8. 生成形状分析对比图...")
    #     create_comparison_visualization(adaptive_shapes, image_path)

        print(f"=== 图像处理完成: {os.path.basename(image_path)} ===")
        print(f"成功处理 {len(processed_rooms)} 个房间的轮廓数据")
        print(f"计算了 {len(room_rectangles)} 个房间的最小外接矩形")
        print(f"处理结果目录: {output_dir}")

        return {
            "cad_rooms": cad_rooms,
            "lighting_rooms": lighting_payload.get("rooms", {}),
        }
    finally:
        if previous_output_dir is None:
            os.environ.pop("CAD_STEP_OUTPUT_DIR", None)
        else:
            os.environ["CAD_STEP_OUTPUT_DIR"] = previous_output_dir


def process_images_batch(image_directory, cad_params=None, save_to_file=True):
    """
    批量处理指定目录下的所有PNG图像
    :param image_directory: 图像目录路径
    :param cad_params: CAD参数字典，如果为None则使用默认参数
    :param save_to_file: 是否保存中间结果到文件
    :return: 批量处理结果字典
    """
    print(f"=== 开始批量处理目录: {image_directory} ===")

    if not os.path.exists(image_directory):
        raise FileNotFoundError(f"指定目录不存在: {image_directory}")

    # 获取目录下所有PNG文件
    png_files = []
    for file in os.listdir(image_directory):
        if file.lower().endswith('.png'):
            png_files.append(os.path.join(image_directory, file))

    if not png_files:
        raise ValueError(f"目录中没有找到PNG文件: {image_directory}")

    print(f"找到 {len(png_files)} 个PNG文件")

    # 处理每个图像
    results = {}
    for i, image_path in enumerate(png_files, 1):
        print(f"\n--- 处理第 {i}/{len(png_files)} 个图像 ---")
        try:
            result = process_single_image(image_path, cad_params, save_to_file)
            image_name = os.path.basename(image_path)
            results[image_name] = result
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            results[os.path.basename(image_path)] = {'error': str(e)}

    print(f"\n=== 批量处理完成，成功处理 {len([r for r in results.values() if 'error' not in r])} 个图像 ===")

    return results


def main():
    """
    主处理流程（兼容性保持）
    """
    # 输入图像路径
    image_path = '/home/chen/punchy/CAD_knowledge_service/images/test_8K.png'

    print("=== CAD图纸房间分析开始 ===")

    result = process_single_image(image_path, save_to_file=True)

    print("\n=== 处理完成 ===")

    return result


if __name__ == "__main__":
    try:
        results = main()
        print("程序执行成功！")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
