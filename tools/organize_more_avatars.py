import os
import shutil

def organize_avatars():
    """
    将 moreavatars/ 目录中的图片，根据自身名称，
    放入 data/avatars 中对应名称的子目录中。
    如果遇到同名文件，则进行重命名。
    """
    source_dir = 'moreavatars'
    dest_base_dir = 'data/avatars'

    # 确保源目录存在
    if not os.path.isdir(source_dir):
        print(f"错误：源目录 '{source_dir}' 不存在。")
        return

    # 确保目标基础目录存在
    os.makedirs(dest_base_dir, exist_ok=True)

    print(f"开始处理 '{source_dir}' 目录中的文件...")

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)

        # 只处理文件
        if os.path.isfile(source_path):
            base_name, extension = os.path.splitext(filename)

            # 创建目标子目录
            dest_subdir = os.path.join(dest_base_dir, base_name)
            os.makedirs(dest_subdir, exist_ok=True)

            # 构造初始目标文件路径
            dest_path = os.path.join(dest_subdir, filename)

            # 检查文件是否已存在，如果存在则重命名
            counter = 1
            while os.path.exists(dest_path):
                new_filename = f"{base_name}_{counter}{extension}"
                dest_path = os.path.join(dest_subdir, new_filename)
                counter += 1
            
            # 移动文件
            try:
                shutil.move(source_path, dest_path)
                if dest_path.endswith(filename):
                    print(f"已将 '{source_path}' 移动到 '{dest_path}'")
                else:
                    print(f"检测到同名文件，已将 '{source_path}' 重命名并移动到 '{dest_path}'")
            except Exception as e:
                print(f"移动 '{source_path}' 时出错: {e}")

    print("\n所有文件处理完毕。")

if __name__ == '__main__':
    organize_avatars()