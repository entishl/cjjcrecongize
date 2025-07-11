import json

def check_coordinate_sizes(file_path):
    """
    检查坐标文件中定义的所有区域的尺寸。
    """
    with open(file_path, 'r') as f:
        regions = json.load(f)

    all_correct_size = True
    for region in regions:
        name = region['name']
        x1, y1, x2, y2 = region['rect']
        width = x2 - x1
        height = y2 - y1
        if width != 82 or height != 82:
            print(f"[WARNING] Region '{name}' has incorrect size: {width}x{height}")
            all_correct_size = False
    
    if all_correct_size:
        print("All regions have the correct size of 82x82.")

if __name__ == "__main__":
    check_coordinate_sizes('data/coordinates.json')