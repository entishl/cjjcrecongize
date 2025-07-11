import os
from PIL import Image
import sys

def crop_avatars_to_square(avatars_path):
    """
    Crops all images in the specified directory to a 1:1 aspect ratio.

    It crops from the right or the bottom to make the image square.
    Images that are already square are skipped.

    Args:
        avatars_path (str): The path to the directory containing avatar images.
    """
    if not os.path.isdir(avatars_path):
        print(f"Error: Directory not found at '{avatars_path}'")
        return

    print(f"Processing images in '{avatars_path}'...")

    processed_files = 0
    skipped_files = 0
    error_files = 0

    for filename in os.listdir(avatars_path):
        file_path = os.path.join(avatars_path, filename)

        if os.path.isdir(file_path):
            continue

        try:
            with Image.open(file_path) as img:
                width, height = img.size

                if width == height:
                    # print(f"Skipping '{filename}' (already 1:1).")
                    skipped_files += 1
                    continue

                if width > height:
                    # Crop the right side
                    left = 0
                    top = 0
                    right = height
                    bottom = height
                    # print(f"Cropping '{filename}' from {width}x{height} to {height}x{height}.")
                else:  # height > width
                    # Crop the bottom side
                    left = 0
                    top = 0
                    right = width
                    bottom = width
                    # print(f"Cropping '{filename}' from {width}x{height} to {width}x{width}.")

                cropped_img = img.crop((left, top, right, bottom))
                # Convert to RGB to avoid issues with saving in different formats like JPEG
                if cropped_img.mode in ("RGBA", "P"):
                    cropped_img = cropped_img.convert("RGB")
                cropped_img.save(file_path)
                processed_files += 1

        except (IOError, SyntaxError):
            # print(f"Could not process '{filename}'. It might not be an image file.")
            error_files += 1
        except Exception as e:
            print(f"An unexpected error occurred with file '{filename}': {e}")
            error_files += 1
    
    print("\n--- Processing Summary ---")
    print(f"Successfully cropped: {processed_files} file(s)")
    print(f"Skipped (already 1:1): {skipped_files} file(s)")
    print(f"Failed or not an image: {error_files} file(s)")
    print("--------------------------")


if __name__ == "__main__":
    # The script is in tools/, so the project root is one level up.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # The target directory is data/avatars relative to the project root
    avatars_dir = os.path.join(project_root, 'data', 'avatars')
    
    crop_avatars_to_square(avatars_dir)