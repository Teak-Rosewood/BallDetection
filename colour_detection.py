import cv2 as cv
import numpy as np
import cvzone 
from cvzone.ColorModule import ColorFinder

capture = cv.VideoCapture(0)
Colour = ColorFinder(True)
hsvVals = {'hmin': 33, 'smin': 84, 'vmin': 56, 'hmax': 50, 'smax': 155, 'vmax': 255}

while True:
    isTrue, frame = capture.read()
    img, mask = Colour.update(frame, hsvVals)
    cv.imshow('Video Capture', frame)
    cv.imshow('Visual', img)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()