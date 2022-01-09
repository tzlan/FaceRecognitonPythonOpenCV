import cv2
import face_recognition_models

img = cv2.imread("ThomasShelby.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition_models