import cv2

# Load the image
img = cv2.imread("DQN/dqn_eval.jpg")
h, w, _ = img.shape

# Set your desired crop size as a ratio of the original
crop_ratio_w = 0.4  # e.g., keep 60% of width
crop_ratio_h = 0.7  # e.g., keep 60% of height

# Compute crop dimensions
crop_w = int(w * crop_ratio_w)
crop_h = int(h * crop_ratio_h)

# Compute top-left corner of the crop to center it
x_start = (w - crop_w) // 2
y_start = (h - crop_h) // 2

# Perform the crop
cropped = img[y_start:y_start + crop_h, x_start:x_start + crop_w]

# Save the cropped image
cv2.imwrite("DQN/cropped_final_frame.jpg", cropped)
