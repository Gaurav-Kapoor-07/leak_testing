import cv2
import numpy as np

# Load grayscale frame
img = cv2.imread("/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_41_37_Pro/leak/00505.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Canny
edges = cv2.Canny(img, 50, 150)

# Extract edge coordinates
edge_points = cv2.findNonZero(edges)
coordinates = [(pt[0][0], pt[0][1]) for pt in edge_points] if edge_points is not None else []

# Optional: Visualize
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Found {len(coordinates)} edge points")
