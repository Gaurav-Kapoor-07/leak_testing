import cv2
import os
import numpy as np

# Path to CLAHE-processed grayscale images
input_folder = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_52_07_Pro/leak_clahe/"

color_input_folder = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_52_07_Pro/leak/"

output_folder = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_52_07_Pro/leak_clahe_diff/"

output_folder_2 = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_52_07_Pro/leak_clahe_diff_2/"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

init_frame = None
frame_count = 0

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
    # _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

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

    np.pi

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # x, y, w, h = cv2.boundingRect(cnt)
        # aspect_ratio = float(w) / h

        # # Final filtering
        # if (
        #     0.75 < aspect_ratio < 1.3 and
        #     0.7 < circularity < 1.2 and
        #     area > 5
        # ):
        #     cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if (0.7 < circularity < 1.2):
            cv2.drawContours(frame_color, [cnt], -1, (0, 255, 0), 2)

    #     cv2.imshow("Thresholded Difference", thresh_clean)
    #     cv2.imshow("Detected Bubbles", frame_color)
    #     cv2.waitKey(int(sleep * 1000))

    # Save the output image
    save_path = os.path.join(output_folder, image_name)
    cv2.imwrite(save_path, frame_color)

    frame_count += 1
