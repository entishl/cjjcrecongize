import gradio as gr
import numpy as np
import pandas as pd
import cv2
import os
from recognizer import load_config, AvatarRecognizer, visualize_results, Config

# 1. 全局加载模型和配置 (应用启动时只执行一次)
print("正在加载配置和模型...")
try:
    config = load_config()
    recognizer = AvatarRecognizer(config)
    print("模型和配置加载成功。")
except Exception as e:
    print(f"错误：模型加载失败，{e}")
    recognizer = None

def recognize_and_visualize(image_array: np.ndarray):
    """
    核心处理函数：接收图像，返回可视化结果和数据帧。
    """
    if recognizer is None:
        raise gr.Error("识别器未能成功初始化，请检查后台日志。")

    print("接收到新的识别请求...")
    # Gradio 传入的图像是 RGB 格式，符合我们内部处理要求
    
    try:
        # 2. 执行识别
        results, win_loss_results, cropped_avatars, processed_image = recognizer.recognize(image_array)
        
        # 3. 生成可视化结果
        print("正在生成可视化图像...")
        output_image = visualize_results(processed_image, results, recognizer.crop_regions)
        
        # 4. 生成 CSV 格式的数据
        print("正在生成数据表格...")
        columns = ["A1", "A2", "A3", "A4", "A5", "WOL", "D1", "D2", "D3", "D4", "D5"]
        results_map = {res['position']: res['character'] for res in results}
        csv_data = []
        for i in range(5):
            team_num = i + 1
            row = {
                "A1": results_map.get(f"P1_T{team_num}_1", "N/A"),
                "A2": results_map.get(f"P1_T{team_num}_2", "N/A"),
                "A3": results_map.get(f"P1_T{team_num}_3", "N/A"),
                "A4": results_map.get(f"P1_T{team_num}_4", "N/A"),
                "A5": results_map.get(f"P1_T{team_num}_5", "N/A"),
                "WOL": win_loss_results[i] if i < len(win_loss_results) else "N/A",
                "D1": results_map.get(f"P2_T{team_num}_1", "N/A"),
                "D2": results_map.get(f"P2_T{team_num}_2", "N/A"),
                "D3": results_map.get(f"P2_T{team_num}_3", "N/A"),
                "D4": results_map.get(f"P2_T{team_num}_4", "N/A"),
                "D5": results_map.get(f"P2_T{team_num}_5", "N/A"),
            }
            csv_data.append(row)
        df = pd.DataFrame(csv_data, columns=columns)

        # 5. 保存识别失败的切片
        print("正在检查并保存识别失败的切片...")
        save_failed_crops(results, cropped_avatars, config.paths.output_dir, "huggingface_upload")

        print("处理完成。")
        return output_image, df

    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        # 返回一个错误信息给用户界面
        raise gr.Error(f"处理图像时发生错误: {e}")

def save_failed_crops(
    results: list,
    cropped_avatars: list,
    output_dir: str,
    original_filename: str
):
    """
    仅保存识别失败的头像切片。
    """
    failed_crops_found = False
    base_name, _ = os.path.splitext(original_filename)
    run_dir = os.path.join(output_dir, f"{base_name}_failed_crops")

    for i, result in enumerate(results):
        if result['character'] == "未知":
            if not failed_crops_found:
                print("检测到识别失败的切片，正在保存...")
                os.makedirs(run_dir, exist_ok=True)
                failed_crops_found = True
            
            position_name = result['position']
            
            image_path = os.path.join(run_dir, f"{position_name}.png")
            img_bgr = cv2.cvtColor(cropped_avatars[i], cv2.COLOR_RGB2BGR)
            is_success, buffer = cv2.imencode(".png", img_bgr)
            if is_success:
                with open(image_path, 'wb') as f:
                    f.write(buffer)

    if failed_crops_found:
        print(f"识别失败的切片已保存至: {run_dir}")


# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# NIKKE 竞技场角色识别器")
    gr.Markdown("上传一张游戏竞技场截图，应用将自动识别双方队伍的角色，并以图像和表格形式展示结果。")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="上传截图")
            submit_btn = gr.Button("开始识别")
        with gr.Column():
            image_output = gr.Image(label="识别结果图")
            dataframe_output = gr.DataFrame(label="详细结果")

    submit_btn.click(
        fn=recognize_and_visualize,
        inputs=image_input,
        outputs=[image_output, dataframe_output]
    )

if __name__ == "__main__":
    demo.launch()