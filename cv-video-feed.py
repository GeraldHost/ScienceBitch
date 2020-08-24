import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces_rects = haar_cascade_face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);

    print('Faces found: ', len(faces_rects))

    for (x,y,w,h) in faces_rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
