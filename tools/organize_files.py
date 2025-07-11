import os
import re
import shutil
import argparse
from collections import defaultdict

def organize_avatar_files(base_dir: str, dry_run: bool):
    """
    自动整理头像文件夹，将角色皮肤变体合并到基础角色文件夹中。

    Args:
        base_dir (str): 包含角色文件夹的根目录 (例如 'data/avatars')。
        dry_run (bool): 如果为 True，则只打印将要执行的操作，不实际移动文件。
    """
    if not os.path.isdir(base_dir):
        print(f"错误: 目录 '{base_dir}' 不存在。")
        return

    print(f"扫描目录: {base_dir}")
    
    # 1. 收集所有源目录并解析基础角色名
    source_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # 使用字典来组织，键是基础角色名，值是源文件夹列表
    character_map = defaultdict(list)
    
    # 正则表达式匹配 "角色名_数字" 格式
    pattern = re.compile(r'(.+?)_(\d+)$')

    for dir_name in source_dirs:
        match = pattern.match(dir_name)
        if match:
            base_name = match.group(1)
            # 处理特殊情况，如 "阿妮斯：闪耀夏日" 被错误分割
            if ':' in base_name and dir_name.startswith(base_name):
                 character_map[dir_name].append(dir_name) # 如果是皮肤名，则不合并
            else:
                 character_map[base_name].append(dir_name)
        else:
            # 如果不匹配模式，则将文件夹本身视为一个基础角色
            character_map[dir_name].append(dir_name)

    print(f"找到 {len(source_dirs)} 个源文件夹，解析出 {len(character_map)} 个基础角色。")
    print("-" * 30)

    # 2. 遍历基础角色，创建目标文件夹并移动文件
    for base_name, source_list in character_map.items():
        # 如果一个基础角色只关联一个文件夹，且文件夹名与基础名相同，则无需操作
        if len(source_list) == 1 and source_list[0] == base_name:
            continue

        target_dir = os.path.join(base_dir, base_name)
        
        print(f"处理基础角色: '{base_name}'")
        print(f"  - 目标文件夹: {target_dir}")
        
        if not os.path.exists(target_dir) and not dry_run:
            os.makedirs(target_dir)
            print(f"  - [创建] {target_dir}")

        for source_name in source_list:
            source_dir_path = os.path.join(base_dir, source_name)
            print(f"  - 源文件夹: {source_dir_path}")
            
            try:
                for filename in os.listdir(source_dir_path):
                    source_file = os.path.join(source_dir_path, filename)
                    target_file = os.path.join(target_dir, filename)

                    # 处理文件名冲突
                    if os.path.exists(target_file):
                        name, ext = os.path.splitext(filename)
                        i = 1
                        while True:
                            new_filename = f"{name}_{i}{ext}"
                            new_target_file = os.path.join(target_dir, new_filename)
                            if not os.path.exists(new_target_file):
                                target_file = new_target_file
                                break
                            i += 1
                        print(f"    - [警告] 文件冲突, '{filename}' 将被重命名为 '{os.path.basename(target_file)}'")

                    print(f"    - 移动 '{source_file}' -> '{target_file}'")
                    if not dry_run:
                        shutil.move(source_file, target_file)
            except FileNotFoundError:
                print(f"    - [警告] 源文件夹 '{source_dir_path}' 在处理过程中消失，可能已被移动。")


        # 3. 删除空的源文件夹 (仅在非dry-run模式下)
        if not dry_run:
            for source_name in source_list:
                # 只有当源文件夹不是目标文件夹时才删除
                if source_name != base_name:
                    source_dir_path = os.path.join(base_dir, source_name)
                    try:
                        if not os.listdir(source_dir_path):
                            os.rmdir(source_dir_path)
                            print(f"  - [删除空目录] {source_dir_path}")
                    except OSError as e:
                        print(f"  - [错误] 删除目录 '{source_dir_path}' 失败: {e}")
        
        print("-" * 30)

    print("文件整理完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="整理Nikke头像文件夹，将皮肤变体合并到基础角色文件夹中。")
    parser.add_argument(
        "path",
        type=str,
        nargs='?',
        default="data/avatars",
        help="头像库的根目录路径 (默认为: 'data/avatars')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="执行预演，只打印将要进行的操作，不实际移动或删除任何文件。"
    )
    
    args = parser.parse_args()

    if args.dry_run:
        print("*** 预演模式 (Dry Run) ***")
        print("将只显示操作，不会对文件系统进行任何更改。\n")

    organize_avatar_files(args.path, args.dry_run)