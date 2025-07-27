import cv2
import os
import time

# Path to folder with frames (images)
folder_path = "/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_41_37_Pro/leak"
image_files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

prev_frame = None
frame_count = 0
N = 1
sleep = 0.1  # seconds
gb = 9     # Gaussian blur kernel size

for image_name in image_files:
    image_path = os.path.join(folder_path, image_name)
    frame = cv2.imread(image_path)
    if frame is None:
        continue

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.medianBlur(frame_gray, 5)
    frame_blur = cv2.GaussianBlur(frame_gray, (gb, gb), 0)

    if frame_count % N == 0:
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame_blur)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            # Morphological Opening (erosion followed by dilation)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.75 < aspect_ratio < 1.3 and cv2.contourArea(cnt) > 50:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show the outputs
            cv2.imshow("Difference", thresh_clean)
            cv2.imshow("Detected", frame)
            cv2.waitKey(int(sleep * 1000))

        prev_frame = frame_blur.copy()

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
