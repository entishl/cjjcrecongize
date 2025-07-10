import os
from PIL import Image

def add_white_background(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            try:
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path).convert("RGBA")

                # 创建一个白色背景的图片
                bg = Image.new("RGBA", img.size, "WHITE")
                # 将原始图片粘贴到背景上
                bg.paste(img, (0, 0), img)

                # 转换为RGB格式进行保存
                bg = bg.convert("RGB")

                output_path = os.path.join(output_dir, filename)
                bg.save(output_path, "PNG")
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Could not process {filename}: {e}")

if __name__ == '__main__':
    input_directory = 'data/avatars'
    output_directory = 'data/avatars_with_background'
    add_white_background(input_directory, output_directory)