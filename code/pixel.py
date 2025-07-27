import cv2

# Load image
img = cv2.imread("/home/gaurav-24/leak_testing/frames/frames_WIN_20250724_17_41_37_Pro/leak_clahe/00505.jpg")

# Coordinates where you want to place the box (top-left corner)
x, y = 50, 50

# Draw a 5x5 pixel rectangle
cv2.rectangle(img, (x, y), (x + 20, y + 20), (0, 0, 255), -1)  # filled red square

# Optional: Add a label
cv2.putText(img, "5x5 px", (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Show image
cv2.imshow("Image with 5x5 pixel box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
