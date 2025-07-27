import cv2
import numpy as np
import os

# Input and output folder paths
input_folder = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_41_37_Pro/leak_clahe/"
output_folder = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_41_37_Pro/leak_clahe_hough/"
output_folder_2 = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_41_37_Pro/leak_clahe_hough_canny/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)

# List and sort image files (assuming .jpg; change if needed)
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])

for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: Failed to read {filename}")
        continue

    img_blur = cv2.medianBlur(img, 5)  # Remove small noise

    edges = cv2.Canny(img_blur, 50, 150)

    # Detect circles
    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=5,
        param1=50,
        param2=30,
        minRadius=2,
        maxRadius=20
    )

    # Draw circles
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)

    # cv2.imshow("Bubbles Detected", output)

    # Save result with same filename
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, output)

    save_path_2 = os.path.join(output_folder_2, filename)
    cv2.imwrite(save_path_2, edges)

print("✅ hough complete.")
