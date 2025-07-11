import os
import json

def rename_files():
    # 定义nikkes.json文件和moreavatars目录的路径
    # 脚本位于tools/下，所以需要回到上一级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'nikkes.json')
    avatars_dir = os.path.join(base_dir, 'moreavatars')

    # 检查路径是否存在
    if not os.path.exists(json_path):
        print(f"错误: 'nikkes.json' not found at {json_path}")
        return
    if not os.path.exists(avatars_dir):
        print(f"错误: 'moreavatars' directory not found at {avatars_dir}")
        return

    # 读取并解析nikkes.json
    with open(json_path, 'r', encoding='utf-8') as f:
        nikkes_data = json.load(f)

    # 创建一个从id到name_cn的映射
    id_to_name_cn = {str(item['id']): item['name_cn'] for item in nikkes_data}

    # 遍历moreavatars目录中的文件
    for filename in os.listdir(avatars_dir):
        # 从文件名中分离出ID和扩展名
        file_id, file_ext = os.path.splitext(filename)

        # 在映射中查找对应的name_cn
        new_name_cn = id_to_name_cn.get(file_id)

        if new_name_cn:
            # 构建新的文件名
            new_filename = f"{new_name_cn}{file_ext}"
            old_filepath = os.path.join(avatars_dir, filename)
            new_filepath = os.path.join(avatars_dir, new_filename)

            # 执行重命名
            try:
                os.rename(old_filepath, new_filepath)
                print(f"重命名: '{filename}' -> '{new_filename}'")
            except OSError as e:
                print(f"重命名 '{filename}' 失败: {e}")
        else:
            print(f"警告: 在 nikkes.json 中未找到ID为 '{file_id}' 的记录，跳过 '{filename}'")

if __name__ == '__main__':
    rename_files()