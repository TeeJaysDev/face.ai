import cv2

trained_face_data = cv2.CascadeClassifier('face.xml')

img = cv2.imread('4.jpg')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coords = trained_face_data.detectMultiScale(grayscaled_img)

(x, y, w, h) = face_coords[0]
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Face Detection', img)
cv2.waitKey()