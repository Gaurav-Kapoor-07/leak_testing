import cv2
import os
import numpy as np

# Path to CLAHE-processed grayscale images
input_folder = "/home/gaurav-24/leak_testing/frames/new/frames_1/leak_clahe/"

color_input_folder = "/home/gaurav-24/leak_testing/frames/new/frames_1/"

output_folder = "/home/gaurav-24/leak_testing/frames/new/frames_1/leak_clahe_diff/"

output_folder_2 = "/home/gaurav-24/leak_testing/frames/new/frames_1/leak_clahe_diff_2/"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

init_frame = None
frame_count = 0

min_area = 10
max_area = 100

grid_size = 100
min_contours_in_cell = 3

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    orig_img_path = os.path.join(color_input_folder, image_name)
    frame_color = cv2.imread(orig_img_path)

    if frame is None or frame_color is None:
        continue

    # Apply median + Gaussian blur to suppress noise
    frame_blur = cv2.medianBlur(frame, 5)
    # frame_blur = cv2.GaussianBlur(frame, (gb, gb), 0)

    # frame_blur = frame

    if init_frame is None:
        init_frame = frame_blur.copy()
        frame_count += 1
        continue  # skip comparing first frame with itself

    diff = cv2.absdiff(init_frame, frame_blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)

    # Morphological Opening (erosion followed by dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    save_path_2 = os.path.join(output_folder_2, image_name)
    cv2.imwrite(save_path_2, diff)

    # Draw all contours in green
    # cv2.drawContours(frame_color, contours, -1, (0, 255, 0), 1)

    # === Store bubble centroids ===
    bubble_centroids = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < min_area or area > max_area or perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if (0.7 < circularity < 1.2):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bubble_centroids.append((cx, cy))
            cv2.drawContours(frame_color, [cnt], -1, (0, 255, 0), 2)

    # === Grid Clustering ===
    height, width = frame_color.shape[:2]
    grid_rows = height // grid_size + 1
    grid_cols = width // grid_size + 1

    # Track number of bubbles per grid cell
    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

    for (cx, cy) in bubble_centroids:
        grid_x = cx // grid_size
        grid_y = cy // grid_size
        grid_counts[grid_y, grid_x] += 1

    # === Draw active grid cells ===
    for y in range(grid_rows):
        for x in range(grid_cols):
            if grid_counts[y, x] > 0:
                top_left = (x * grid_size, y * grid_size)
                bottom_right = ((x + 1) * grid_size, (y + 1) * grid_size)
                cv2.rectangle(frame_color, top_left, bottom_right, (255, 0, 0), 2)
                cv2.putText(frame_color, str(grid_counts[y, x]),
                            (top_left[0] + 5, top_left[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    #     cv2.imshow("Thresholded Difference", thresh_clean)
    #     cv2.imshow("Detected Bubbles", frame_color)
    #     cv2.waitKey(int(sleep * 1000))

    # Save the output image
    save_path = os.path.join(output_folder, image_name)
    cv2.imwrite(save_path, frame_color)

    frame_count += 1
