#! /usr/bin/env python
import cv2
from collections import Counter

FONT = cv2.FONT_HERSHEY_COMPLEX

def start_capture(num_samples):
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
        
        if trigger:
            roi = frame[5: box_size-5, width-box_size + 5: width-5]
            counter[trigger] += 1
            char = chr(trigger)
            filename = "images/%s/%s-%s.jpg" % (char, char, counter[trigger])
            cv2.imwrite(filename, roi) 
            capture_text = "Collected Samples of %s: %s" % (chr(trigger), counter[trigger])
            cv2.putText(frame, capture_text, (3,300), FONT, 0.45, (0,0,255), 1, cv2.LINE_AA)

        prompt_text = "Press 'r' to collect rock samples, 'p' for paper, 's' for scissor and 'n' for nothing"
        cv2.putText(frame, prompt_text, (3,350), FONT, 0.45, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        
        k = cv2.waitKey(1)
        # r: rock, p: paper, s: scissors, n: nothing
        keys = list(map(lambda x: ord(x), ['r', 'p', 's', 'n']))
        if k in keys:
            trigger = k
        
        if k == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    no_of_samples = 500
    start_capture(no_of_samples)

