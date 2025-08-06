import cv2
import os
import zipfile

# Paths
video_path = "/home/gaurav-24/leak_testing/videos/2.mp4"
output_folder = "/home/gaurav-24/leak_testing/frames/new/frames_2/"
zip_name = "cvat_ready_images.zip"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f"{frame_idx:05d}.jpg")
    cv2.imwrite(filename, frame)
    frame_idx += 1

cap.release()
print(f"Extracted {frame_idx} frames.")

# # Zip the folder
# with zipfile.ZipFile(zip_name, 'w') as zipf:
#     for root, _, files in os.walk(output_folder):
#         for file in sorted(files):
#             file_path = os.path.join(root, file)
#             arcname = os.path.relpath(file_path, output_folder)
#             zipf.write(file_path, arcname)

# print(f"Zipped into: {zip_name}")
