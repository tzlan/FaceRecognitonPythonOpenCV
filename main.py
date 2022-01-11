import cv2
import face_recognition




img = cv2.imread("Images/ThomasShelby.webp")
rgb_img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]


img2 = cv2.imread("Images/Tony-Stark.webp")
rgb_img2 = cv2.cvtColor(img2 , cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]



result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.imwrite("Img.png" , img)
cv2.imwrite("Img2.png" , img2)

cv2.waitKey(0)






