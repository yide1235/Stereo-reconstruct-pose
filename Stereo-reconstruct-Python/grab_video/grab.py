


import cv2

# Open the first video
video1 = cv2.VideoCapture('./video/left.mp4')

# Open the second video
video2 = cv2.VideoCapture('./video/right.mp4')

# Check if both videos are opened successfully
if not video1.isOpened() or not video2.isOpened():
    print("Error opening videos")
    exit()

# Frame counter
frame_counter = 0

while True:
    # Read frame from both videos
    ret1, frame_from_video1 = video1.read()
    ret2, frame_from_video2 = video2.read()

    # Check if frames are read successfully
    if not ret1 or not ret2:
        print("Finished reading frames.")
        break

    # Save the frames as images
    cv2.imwrite(f'./left/frame{frame_counter}.jpg', frame_from_video1)
    cv2.imwrite(f'./right/frame{frame_counter}.jpg', frame_from_video2)

    # Increment the frame counter
    frame_counter += 1

# Release the video captures
video1.release()
video2.release()