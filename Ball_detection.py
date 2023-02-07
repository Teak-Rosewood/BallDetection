from cmath import pi
import cv2 as cv
from cv2 import Laplacian
import numpy as np

capture = cv.VideoCapture(5)

# Setting min and max values for HSV 


# Setting min and max values for BGR

bgrVals =  {'hmin': 21, 'smin': 78, 'vmin': 84, 'hmax': 35, 'smax': 255, 'vmax': 255}
# {'hmin': 28, 'smin': 85, 'vmin': 66, 'hmax': 45, 'smax': 255, 'vmax': 222}
lower_lim_hsv  = np.array([bgrVals['hmin'], bgrVals['smin'], bgrVals['vmin']])
upper_lim_hsv  = np.array([bgrVals['hmax'], bgrVals['smax'], bgrVals['vmax']])

while True:
    isTrue, frame = capture.read()

    # Applying blur
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    blurred = cv.medianBlur(blurred, 5, 0)
    

    # Converting to HSV
    frame_hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    frame_HLS = cv.cvtColor(blurred, cv.COLOR_BGR2HLS)

    #C onfiguring and filtering bitmask
    mask = cv.inRange(frame_hsv, lower_lim_hsv, upper_lim_hsv)
    mask = cv.GaussianBlur(mask, (9, 9), 20)
    # mask = cv.erode(mask, (3, 3), iterations=1)
    # mask = cv.dilate(mask, (3, 3), iterations=1)
    hsv_mask = cv.bitwise_and(frame_hsv, frame_hsv, mask = mask)
    
    # converting hsv_mask to grayscale 
    grayscale = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)
    grayscale = cv.cvtColor(grayscale, cv.COLOR_BGR2GRAY)
    # Edge detection on HSV mask
    sobelx = cv.Sobel(grayscale, cv.CV_64F, 2, 0)
    sobely = cv.Sobel(grayscale, cv.CV_64F, 0, 2)
    sobel_edges = cv.bitwise_or(sobelx, sobely)
    sobel_edges = cv.bitwise_not(sobel_edges)

    #hsv_mask = cv.bitwise_and(hsv_mask, hsv_mask, mask = grayscale)

    # USING HOUGH CIRCLES 

    #converting HSV to grayscale 
    circle_detection = cv.HoughCircles(mask,cv.HOUGH_GRADIENT, 1.85  , 100, minRadius= 12)
                                    # param1=380,param2=120,minRadius=6 ,maxRadius= 300)
    if circle_detection is not None:
        circle_detection = np.uint16(np.around(circle_detection))
        for circle in circle_detection[0, :]:
            a, b, r = circle[0], circle[1], circle[2]
            cv.circle(frame, (a, b), r, (255, 0, 0), 2)
            cv.circle(frame, (a, b), 1, (255, 0, 0), 3)

    # # Using Contours 

    # #finding contours and sorting 
    # contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contours = sorted(contours, key=lambda x: cv.contourArea(x))

    # # Drawing around contours
    # for contour in contours:
    #     area = cv.contourArea(contour)
    #     ((circle_x, circle_y), radius) = cv.minEnclosingCircle(contour)
    #     true_area = pi * radius * radius
    #     if area > 100:
    #         if radius > 14 and area >= true_area * 3 / 5 and area <= true_area * 5 / 4:
    #             cv.circle(frame, (int(circle_x), int(circle_y)), 3, (255, 0, 0), cv.FILLED)
    #             cv.circle(frame, (int(circle_x), int(circle_y)), int(radius), (255, 0, 0), 2)

    cv.imshow('Video Feed', frame)
    cv.imshow('HSV Mask', mask)
    # cv.imshow('Sobel Edge Detection', sobel_edges)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
    
    