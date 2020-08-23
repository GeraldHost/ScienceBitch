import cv2
import matplotlib.pyplot as plt

def convertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

IM_FILE='women-face.jpg'

test_image = cv2.imread(IM_FILE)
test_image_gray = convertToGray(test_image)

haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

print('Faces found: ', len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

while True:
    cv2.imshow('baby', test_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
