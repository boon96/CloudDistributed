# Read Video Frame by Frame, Save frame as image if faces are present

import cv2
# from retinaface import RetinaFace


# Read the video
capture = cv2.VideoCapture("Get_Images.mp4")

# Split Video into individual Frames
counter = 0
while capture.isOpened():
    # Reads the Video Frame By Frame
    ret, frame = capture.read()
    # ret Signifies the status of the read, False if failed to read frame
    if ret:
        # save frame as Jpg
        cv2.imwrite(f"Images/Image_{counter}.jpg",frame)
        counter += 1
        print(f"{counter}")

