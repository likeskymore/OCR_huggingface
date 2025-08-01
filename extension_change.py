import os
from PIL import Image  # Make sure Pillow is installed: pip install pillow

# === Set your directory ===
directory ="dataset"  # ← Replace this with your actual path

# === 1. Rename .gt.txt files back to .txt ===
for filename in os.listdir(directory):
    if filename.endswith(".gt.txt"):
        # Remove only the '.gt' part
        new_filename = filename.replace(".gt.txt", ".txt")
        os.rename(
            os.path.join(directory, filename),
            os.path.join(directory, new_filename)
        )
        print(f"Renamed TXT: {filename} → {new_filename}")
        
# # === 2. Rename .txt files to .gt.txt ===
# for filename in os.listdir(directory):
#     if filename.endswith(".txt") and not filename.endswith(".gt.txt"):
#         base = filename[:-4]  # remove ".txt"
#         new_filename = f"{base}.gt.txt"
#         os.rename(
#             os.path.join(directory, filename),
#             os.path.join(directory, new_filename)
#         )
#         print(f"Renamed TXT: {filename} → {new_filename}")

# === 3. Convert all non-PNG images to PNG ===
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    name, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext in [".jpg", ".jpeg", ".bmp", ".tiff"] and not filename.startswith("."):
        try:
            with Image.open(filepath) as img:
                new_path = os.path.join(directory, f"{name}.png")
                img.convert("RGBA").save(new_path)
            os.remove(filepath)
            print(f"Converted: {filename} → {name}.png")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
