import cv2
import os

# Define the path to your MP4 video
video_path = "../random_agent.mp4"

# Output image path
output_image_path = "random/random_agent.jpg"

# Open the video
cap = cv2.VideoCapture(video_path)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set to the last frame
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

# Read the last frame
ret, frame = cap.read()

# Save the frame as an image if successful
if ret:
    cv2.imwrite(output_image_path, frame)
    result = f"Final frame saved to {output_image_path}"
else:
    result = "Failed to read the last frame"

cap.release()

