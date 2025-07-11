import os
from PIL import Image

def process_images():
    input_dir = 'tobefinished/'
    output_dir = 'moreavatars/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            try:
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path).convert("RGBA")

                # 1. 裁去顶部的63像素
                width, height = img.size
                top_crop = 63
                img_cropped_top = img.crop((0, top_crop, width, height))

                # 2. 裁去底部，使图片为1:1
                new_width, new_height = img_cropped_top.size
                if new_width != new_height:
                    crop_size = new_width
                    img_square = img_cropped_top.crop((0, 0, crop_size, crop_size))
                else:
                    img_square = img_cropped_top

                # 3. 创建白色背景
                background = Image.new('RGB', img_square.size, (255, 255, 255))
                
                # 4. 将透明图片粘贴到白色背景上
                background.paste(img_square, (0, 0), img_square)

                # 5. 保存图片
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_dir, output_filename)
                background.save(output_path, 'PNG')
                print(f"Processed {filename} and saved to {output_path}")

            except Exception as e:
                print(f"Could not process {filename}: {e}")

if __name__ == "__main__":
    process_images()