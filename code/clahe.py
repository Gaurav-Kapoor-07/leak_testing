import cv2
import os

# Input and output folder paths
input_folder = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_52_07_Pro/leak/"
output_folder = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_52_07_Pro/leak_clahe/"
# output_folder = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_52_07_Pro/leak_clahe_10/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List and sort image files (assuming .jpg; change if needed)
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])

# CLAHE setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
# clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Failed to read {filename}")
        continue

    # Convert to grayscale (CLAHE works best on 1 channel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_img = clahe.apply(gray)

    # Save result with same filename
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, clahe_img)

print("✅ CLAHE preprocessing complete.")
