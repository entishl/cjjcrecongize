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
    output_dir: str
):
    """
    将裁剪的头像和识别分数保存到各自的目录中。

    Args:
        results (List[Dict[str, Any]]): 识别结果列表。
        cropped_avatars (List[np.ndarray]): 裁剪的头像图像列表。
        output_dir (str): 保存结果的根目录。
    """
    print("\n正在保存裁剪的头像和分数...")
    # 为这个输入图像创建一个唯一的子目录
    run_dir = os.path.join(output_dir, "recognition_details")
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
                        help="待识别的游戏截图路径。")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="是否显示并保存可视化结果。")
    parser.add_argument("--save-crops", action="store_true",
                        help="是否将被识别的头像切片保存到目录中。")
    args = parser.parse_args()

    try:
        # 1. 加载配置和初始化识别器
        print("正在加载配置和模型...")
        config = load_config()
        recognizer = AvatarRecognizer(config)
        print("初始化完成。")

        # 2. 执行识别
        print(f"\n正在识别图像: {args.input}")
        results, cropped_avatars = recognizer.recognize(args.input)

        # 3. 打印结果
        print("\n识别结果:")
        pprint.pprint(results)

        # 4. 可选：保存切片
        if args.save_crops:
            save_cropped_results(results, cropped_avatars, config.paths.output_dir)

        # 5. 可选：可视化
        if args.visualize:
            print("\n正在生成可视化结果...")
            output_image = visualize_results(args.input, results, recognizer.crop_regions)
            
            # 保存结果
            output_dir = config.paths.output_dir
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(args.input)
            file_name, file_ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{file_name}_result{file_ext}")
            
            cv2.imwrite(output_path, output_image)
            print(f"可视化结果已保存至: {output_path}")

            # 显示结果
            cv2.imshow("Recognition Result", output_image)
            print("按任意键退出显示...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保文件路径正确，并且特征数据库已生成。")
    except Exception as e:
        print(f"\n发生了一个意外错误: {e}")

if __name__ == "__main__":
    main()