from cmath import pi
import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

# Setting min and max values for HSV 

hsvVals =  {'hmin': 24, 'smin': 88, 'vmin': 63, 'hmax': 43, 'smax': 186, 'vmax': 255}
# {'hmin': 28, 'smin': 85, 'vmin': 66, 'hmax': 45, 'smax': 255, 'vmax': 222}
lower_lim_hsv  = np.array([hsvVals['hmin'], hsvVals['smin'], hsvVals['vmin']])
upper_lim_hsv  = np.array([hsvVals['hmax'], hsvVals['smax'], hsvVals['vmax']])

# Setting min and max values for BGR

labVals = {'hmin': 57, 'smin': 100, 'vmin': 140, 'hmax': 255, 'smax': 123, 'vmax': 255} 
#{'hmin': 213, 'smin': 91, 'vmin': 125, 'hmax': 255, 'smax': 138, 'vmax': 200}
# {'hmin': 28, 'smin': 85, 'vmin': 66, 'hmax': 45, 'smax': 255, 'vmax': 222}
lower_lim_lab  = np.array([labVals['hmin'], labVals['smin'], labVals['vmin']])
upper_lim_lab  = np.array([labVals['hmax'], labVals['smax'], labVals['vmax']])

while True:
    isTrue, frame = capture.read()

    # Applying blur

    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    blurred = cv.medianBlur(blurred, 5, 0)

    # Converting to HSV

    frame_hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    frame_lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)

    # Configuring and filtering bitmask

    mask_lab = cv.inRange(frame_lab, lower_lim_lab, upper_lim_lab)
    mask_hsv = cv.inRange(frame_hsv, lower_lim_hsv, upper_lim_hsv)
    mask = cv.bitwise_or(mask_hsv, mask_hsv, mask = mask_lab)

    mask = cv.erode(mask, (3, 3), iterations=1)
    mask = cv.dilate(mask, (3, 3), iterations=1)

    hsv_mask = cv.bitwise_and(frame_lab, frame_lab, mask=mask)

    # Converting hsv_mask to grayscale

    # grayscale = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)
    grayscale = cv.cvtColor(hsv_mask, cv.COLOR_BGR2GRAY)

    # Edge detection on HSV mask

    canny_edges = cv.Canny(grayscale, 90, 91)
    canny_edges = cv.bitwise_not(canny_edges)

    # Adding the inverted edges to the mask

    mask = cv.bitwise_and(mask, mask, mask=canny_edges)

    # finding contours and sorting
    contours, hierarchy = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x))

    # Drawing around contours
    for contour in contours:

        # Finding the area of the contour
        area = cv.contourArea(contour)

        # Finding the radius and center of the circle
        ((circle_x, circle_y), radius) = cv.minEnclosingCircle(contour)
        true_area = pi * radius * radius

        if area > 100 and radius > 14 and area >= true_area * 3 / 5.5 and area <= true_area * 5 / 4:
            # Drawing the perimeter of the circle
            cv.circle(frame, (int(circle_x), int(circle_y)),
                      3, (255, 0, 0), cv.FILLED)
            # Marking the center of the circle
            cv.circle(frame, (int(circle_x), int(circle_y)),
                      int(radius), (255, 0, 0), 2)
            # Adding tet to the frame if a ball is detected
            cv.putText(frame, 'Tennis Ball', (int(circle_x) - int(radius), int(circle_y) + int(radius) + 20),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 0, 0), 1)
    
    # Concatenating and displaying output frames 
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    output = np.concatenate((frame, mask), axis = 1)
    cv.imshow('Video Feed and Bitmask', output)
    cv.imshow('Edges', canny_edges)
    cv.imshow('LAB Mask', mask_lab)
    cv.imshow('HSV Mask', mask_hsv)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release the camera and destroy all the windows
capture.release()
cv.destroyAllWindows()
