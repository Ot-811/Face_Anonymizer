import cv2

image_path = './data/face1.jpg'
img  = cv2.imread(image_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]

    blurred_face = cv2.GaussianBlur(face, (99, 99), 20)

    img[y:y+h, x:x+w] = blurred_face

cv2.imshow("Anonymized Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()