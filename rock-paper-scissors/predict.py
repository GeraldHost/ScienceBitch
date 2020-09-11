#! /usr/bin/env python
import cv2
from collections import Counter


#! /usr/bin/env python
import tensorflow as tf
import numpy as np
from tensorflow import keras

img_height = 224
img_width = 224
class_names = ['n', 'p', 'r', 's']
class_name_map = {"n": "nothing", "p": "paper", "r": "rock", "s": "scissors"}


model = tf.keras.models.load_model('model.h5')

model.summary()


FONT = cv2.FONT_HERSHEY_COMPLEX

def start_capture():
    cap = cv2.VideoCapture(0)
    trigger = False
    counter = Counter()
    box_size = 500
    width = int(cap.get(3))
    rect_start = (width - box_size, 0)
    rect_end = (width, box_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)

        if trigger and counter[trigger] == num_samples:
            trigger = False 
        
        cv2.rectangle(frame, rect_start, rect_end, (0, 250, 150), 2)
        roi = frame[5: box_size-5, width-box_size + 5: width-5]
        roi = tf.image.resize(roi, [224, 224])
        roi = tf.expand_dims(roi, 0) # Create a batch
        predictions = model.predict(roi)
        score = tf.nn.softmax(predictions[0])
    
        pred = class_names[np.argmax(score)]
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(pred, 100 * np.max(score))
        )
        capture_text = "Prediction %s" % class_name_map[pred]
        cv2.putText(frame, capture_text, (3,300), FONT, 0.45, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1) 
        if k == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_capture()
