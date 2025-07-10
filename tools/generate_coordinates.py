import json
import os
import shutil
import argparse

def generate_coordinates(x1, y1, n):
    """
    Generates a list of coordinates based on the initial point and side length.
    """
    coordinates = []
    h_step = 141
    v_step = 286
    p2_offset = 1102

    # Generate P1 coordinates
    for i in range(5):
        for j in range(5):
            name = f"P1_T{i+1}_{j+1}"
            rect_x1 = x1 + j * h_step
            rect_y1 = y1 + i * v_step
            rect_x2 = rect_x1 + n
            rect_y2 = rect_y1 + n
            rect = [rect_x1, rect_y1, rect_x2, rect_y2]
            coordinates.append({"name": name, "rect": rect})

    # Generate P2 coordinates
    x1_p2 = x1 + p2_offset
    for i in range(5):
        for j in range(5):
            name = f"P2_T{i+1}_{j+1}"
            rect_x1 = x1_p2 + j * h_step
            rect_y1 = y1 + i * v_step
            rect_x2 = rect_x1 + n
            rect_y2 = rect_y1 + n
            rect = [rect_x1, rect_y1, rect_x2, rect_y2]
            coordinates.append({"name": name, "rect": rect})
            
    return coordinates

def main():
    """
    Main function to generate and save coordinates.
    """
    parser = argparse.ArgumentParser(description="Generate coordinate files.")
    parser.add_argument("x1", type=int, help="Initial x coordinate")
    parser.add_argument("y1", type=int, help="Initial y coordinate")
    parser.add_argument("n", type=int, help="Side length of the rectangle")
    args = parser.parse_args()

    x1 = args.x1
    y1 = args.y1
    n = args.n

    # Define file paths
    # The script is in tools/, so we go up one level to the project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "data", "coordinates.json")
    backup_path = json_path + ".bak"

    # Backup the existing file
    if os.path.exists(json_path):
        shutil.copyfile(json_path, backup_path)
        print(f"Backup of {json_path} created at {backup_path}")

    # Generate new coordinates
    new_coords = generate_coordinates(x1, y1, n)

    # Write the new coordinates to the file
    with open(json_path, "w") as f:
        json.dump(new_coords, f, indent=2)
    
    print(f"Successfully generated and saved new coordinates to {json_path}")

if __name__ == "__main__":
    main()