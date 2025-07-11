import os
import argparse
import pprint
import cv2
import numpy as np
from typing import List, Dict, Any

from recognizer import load_config, AvatarRecognizer, visualize_results, Config

def save_cropped_results(
    results: List[Dict[str, Any]],
    cropped_avatars: List[np.ndarray],
    output_dir: str,
    original_filename: str
):
    """
    将裁剪的头像和识别分数保存到各自的目录中。

    Args:
        results (List[Dict[str, Any]]): 识别结果列表。
        cropped_avatars (List[np.ndarray]): 裁剪的头像图像列表。
        output_dir (str): 保存结果的根目录。
        original_filename (str): 原始输入图像的文件名，用于创建唯一的子目录。
    """
    print("正在保存裁剪的头像和分数...")
    # 为这个输入图像创建一个唯一的子目录
    base_name, _ = os.path.splitext(original_filename)
    run_dir = os.path.join(output_dir, f"{base_name}_details")
    os.makedirs(run_dir, exist_ok=True)

    for i, result in enumerate(results):
        char_name = result['character']
        position_name = result['position']
        
        # 替换文件名中不支持的字符
        safe_char_name = char_name.replace(":", "_").replace("/", "_")
        
        # 根据新的命名规则创建目录
        dir_name = f"{position_name}_{safe_char_name}"
        save_dir = os.path.join(run_dir, dir_name)
        
        os.makedirs(save_dir, exist_ok=True)

        # 保存裁剪的图像，使用 imencode 来避免非 ASCII 路径问题
        image_path = os.path.join(save_dir, "avatar.png")
        # 将 RGB 转换回 BGR
        img_bgr = cv2.cvtColor(cropped_avatars[i], cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", img_bgr)
        if is_success:
            with open(image_path, 'wb') as f:
                f.write(buffer)

        # 保存分数详情
        score_path = os.path.join(save_dir, "scores.txt")
        with open(score_path, 'w', encoding='utf-8') as f:
            f.write(f"Position: {position_name}\n")
            f.write(f"Recognized Character: {char_name}\n")
            f.write(f"Best Score: {result['similarity']:.4f}\n\n")
            f.write("--- All Scores ---\n")
            for char, score in result['all_scores'].items():
                f.write(f"{char}: {score:.4f}\n")
    
    print(f"裁剪结果已保存至: {run_dir}")


def main():
    """
    主函数，程序的入口点。
    """
    parser = argparse.ArgumentParser(description="游戏角色头像识别器")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="待识别的游戏截图文件或目录路径。")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="是否显示并保存可视化结果。")
    parser.add_argument("--save-crops", action="store_true",
                        help="是否将被识别的头像切片保存到目录中。")
    args = parser.parse_args()

    # 1. 确定要处理的文件列表
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"错误: 输入路径 '{input_path}' 不存在。")
        return

    image_files = []
    if os.path.isfile(input_path):
        image_files.append(input_path)
    elif os.path.isdir(input_path):
        print(f"检测到目录输入，将扫描支持的图像文件...")
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                image_files.append(os.path.join(input_path, filename))
        print(f"找到 {len(image_files)} 个图像文件。")

    if not image_files:
        print("错误: 未在指定路径找到可识别的图像文件。")
        return

    try:
        # 2. 加载配置和初始化识别器 (仅一次)
        print("正在加载配置和模型...")
        config = load_config()
        recognizer = AvatarRecognizer(config)
        print("初始化完成。")

        # 3. 循环处理所有文件
        for image_path in image_files:
            print(f"\n--- 正在处理: {os.path.basename(image_path)} ---")
            try:
                # 3.1. 执行识别
                results, cropped_avatars, processed_image = recognizer.recognize(image_path)

                # 3.2. 打印结果
                print("识别结果:")
                pprint.pprint(results)

                # 3.3. 可选：保存切片
                if args.save_crops:
                    original_filename = os.path.basename(image_path)
                    save_cropped_results(results, cropped_avatars, config.paths.output_dir, original_filename)

                # 3.4. 可选：可视化
                if args.visualize:
                    print("正在生成可视化结果...")
                    # 使用处理后的图像进行可视化
                    output_image = visualize_results(processed_image, results, recognizer.crop_regions)
                    
                    # 保存结果
                    output_dir = config.paths.output_dir
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.basename(image_path)
                    file_name, file_ext = os.path.splitext(base_name)
                    output_path = os.path.join(output_dir, f"{file_name}_result{file_ext}")
                    
                    # 使用 imencode 避免非 ASCII 路径问题
                    is_success, buffer = cv2.imencode(file_ext, output_image)
                    if is_success:
                        with open(output_path, 'wb') as f:
                            f.write(buffer)
                        print(f"可视化结果已保存至: {output_path}")
                    else:
                        print(f"错误: 无法保存可视化结果到 {output_path}")

                    # 在批量模式下，不自动显示图片，避免弹出多个窗口
                    if len(image_files) == 1:
                        cv2.imshow("Recognition Result", output_image)
                        print("按任意键退出显示...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

            except FileNotFoundError as e:
                print(f"错误: {e}")
            except Exception as e:
                print(f"处理 {os.path.basename(image_path)} 时发生意外错误: {e}")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保文件路径正确，并且特征数据库已生成。")
    except Exception as e:
        print(f"\n发生了一个意外错误: {e}")

if __name__ == "__main__":
    main()