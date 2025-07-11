import json
import shutil
from datetime import datetime

def backup_file(file_path):
    """备份原始坐标文件"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    try:
        shutil.copyfile(file_path, backup_path)
        print(f"已备份原始文件到: {backup_path}")
    except FileNotFoundError:
        print(f"警告: 未找到原始文件 {file_path}，无需备份。")
        pass

def get_user_input():
    """获取用户输入的第一组坐标"""
    while True:
        try:
            coords_str = input("请输入 P1_T1_1 的四个坐标值 (x1 y1 x2 y2)，以空格分隔: ")
            coords = [int(c) for c in coords_str.split()]
            if len(coords) != 4:
                raise ValueError("必须输入四个整数值。")
            return coords
        except ValueError as e:
            print(f"输入无效: {e}，请重试。")

def generate_coordinates(p1_t1_1_rect):
    """根据初始坐标和规律生成所有坐标"""
    
    # 从 P1_T1_1 计算基础尺寸和常量
    x1, y1, x2, y2 = p1_t1_1_rect
    rect_width = x2 - x1
    rect_height = y2 - y1

    # 根据文件分析得出的固定步长
    x_stride = 141
    y_stride = 286
    p2_x_offset = 1102
    
    num_teams = 5
    num_members_per_team = 5

    all_coords = []

    # 生成 P1 和 P2 的坐标
    for player_idx in range(1, 3): # 1 for P1, 2 for P2
        player_name = f"P{player_idx}"
        
        # 计算当前玩家的起始X坐标
        start_x1_p = x1 + (player_idx - 1) * p2_x_offset
        
        for team_idx in range(1, num_teams + 1):
            # 计算当前队伍的起始Y坐标
            start_y1_t = y1 + (team_idx - 1) * y_stride
            
            for member_idx in range(1, num_members_per_team + 1):
                coord_name = f"{player_name}_T{team_idx}_{member_idx}"
                
                # 计算当前成员的坐标
                current_x1 = start_x1_p + (member_idx - 1) * x_stride
                current_y1 = start_y1_t
                current_x2 = current_x1 + rect_width
                current_y2 = current_y1 + rect_height
                
                all_coords.append({
                    "name": coord_name,
                    "rect": [current_x1, current_y1, current_x2, current_y2]
                })

    return all_coords

def main():
    """主函数"""
    coords_file_path = 'data/coordinates.json'
    
    # 1. 备份文件
    backup_file(coords_file_path)
    
    # 2. 获取用户输入
    p1_t1_1_coords = get_user_input()
    
    # 3. 生成新坐标
    print("正在根据输入计算新坐标...")
    new_coordinates = generate_coordinates(p1_t1_1_coords)
    
    # 4. 写入新文件
    try:
        with open(coords_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_coordinates, f, indent=4)
        print(f"成功！新的坐标已写入 {coords_file_path}")
    except IOError as e:
        print(f"错误: 无法写入文件 {coords_file_path}。 {e}")

if __name__ == '__main__':
    main()