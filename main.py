import os
import argparse
import pprint
import cv2

from recognizer import load_config, AvatarRecognizer, visualize_results

def main():
    """
    主函数，程序的入口点。
    """
    parser = argparse.ArgumentParser(description="游戏角色头像识别器")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="待识别的游戏截图路径。")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="是否显示并保存可视化结果。")
    args = parser.parse_args()

    try:
        # 1. 加载配置和初始化识别器
        print("正在加载配置和模型...")
        config = load_config()
        recognizer = AvatarRecognizer(config)
        print("初始化完成。")

        # 2. 执行识别
        print(f"\n正在识别图像: {args.input}")
        results = recognizer.recognize(args.input)

        # 3. 打印结果
        print("\n识别结果:")
        pprint.pprint(results)

        # 4. 可选：可视化
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