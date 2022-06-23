import cv2

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()