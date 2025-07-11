import json
import os

def generate_coordinates():
    """
    Generates coordinates for two 5x5 grids and saves them to a JSON file.
    """
    coordinates = []
    # Construct the absolute path for the output file
    # The script is in tools/, data/ is in the parent directory
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'coordinates.json')

    # Grid 1 (P1)
    # Initial rectangle: (36, 359, 118, 441)
    p1_x1_start, p1_y1_start = 36, 359
    width = 118 - 36
    height = 441 - 359
    h_step = 141
    v_step = 286

    for row in range(1, 6):
        for col in range(1, 6):
            name = f"P1_T{row}_{col}"
            x1 = p1_x1_start + (col - 1) * h_step
            y1 = p1_y1_start + (row - 1) * v_step
            x2 = x1 + width
            y2 = y1 + height
            coordinates.append({"name": name, "rect": [x1, y1, x2, y2]})

    # Grid 2 (P2)
    # Initial rectangle: (1138, 359, 1220, 441)
    p2_x1_start, p2_y1_start = 1138, 359

    for row in range(1, 6):
        for col in range(1, 6):
            name = f"P2_T{row}_{col}"
            x1 = p2_x1_start + (col - 1) * h_step
            y1 = p2_y1_start + (row - 1) * v_step
            x2 = x1 + width
            y2 = y1 + height
            coordinates.append({"name": name, "rect": [x1, y1, x2, y2]})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coordinates, f, indent=4, ensure_ascii=False)

    print(f"Successfully generated coordinates file at: {output_path}")

if __name__ == "__main__":
    generate_coordinates()