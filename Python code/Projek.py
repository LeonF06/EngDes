# Import dependencies
from ctypes import sizeof
from math import pi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils

# Edge detection function
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 10, 250)
    return edged

# Ball detection function
def is_ball(contour):
    area = cv2.contourArea(contour)
    if (area > 5000 and len(contour) > 50):
        return True
    else:
        return False

# Star detection function
def is_star(contour):
    if (len(contour) > 1):
        return True
    else:
        return False


# Contour detection function
def contour_detection(image, frame):
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    ball_screenCnt = None
    star_screenCnt = None
    i = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.001*peri, True)


        # Draw the star on the frame
        #approx = cv2.approxPolyDP(c, 0.01*peri, True)
        if is_star(c):
            cv2.drawContours(frame, [approx], 0, (0, 0, 255), 3)
            star_screenCnt = approx
        else:
            star_screenCnt = None
        # Draw the ball on the frame
        #approx = cv2.approxPolyDP(c, 0.001*peri, True)
        if is_ball(c):
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            ball_screenCnt = approx
            return frame, ball_screenCnt, star_screenCnt
        else:
            return frame, None, star_screenCnt
        
        

            
        '''cv2.drawContours(frame, [approx], 0, (0, 0, 255), 3) 
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        i = i + 1
        if i == 1:
            cv2.putText(frame, "1", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 2:
            cv2.putText(frame, "2", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 3:
            cv2.putText(frame, "3", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 4:
            cv2.putText(frame, "4", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 5:
            cv2.putText(frame, "5", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 6:
            cv2.putText(frame, "6", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 7:
            cv2.putText(frame, "7", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 8:
            cv2.putText(frame, "8", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 9:
            cv2.putText(frame, "9", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 10:
            cv2.putText(frame, "10", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 11:
            cv2.putText(frame, "11", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        elif i == 12:
            cv2.putText(frame, "12", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))
        elif i == 13:
            cv2.putText(frame, "13", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))
        elif i == 14:
            cv2.putText(frame, "14", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))
        elif i == 15:
            cv2.putText(frame, "15", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))
        elif i == 16:
            cv2.putText(frame, "16", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))
        elif i == 17:
            cv2.putText(frame, "17", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))
        elif i == 18:
            cv2.putText(frame, "18", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))
        elif i == 19:
            cv2.putText(frame, "19", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))
        elif i == 20:
            cv2.putText(frame, "20", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            print(len(approx))'''
        
        '''if len(approx) == 3:
            cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
        elif len(approx) == 4:
            cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
        elif len(approx) == 5: 
            cv2.putText(frame, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
        if len(approx) == 10:
            cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
        else:
            cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))'''
    

# Mask function
def mask(image, screenCnt):
    mask = np.zeros(image.shape, dtype=np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, (255, 255, 255), -1, )
    new_image = cv2.bitwise_and(image, mask)
    return new_image

# Extratct text function
def extract_text(image):
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.recognize(image)
    return result


# Connect the webcam to the computer
cap = cv2.VideoCapture(1)

ret, frame = cap.read()
edge_frame = edge_detection(frame)
contour_frame, ball_screenCnt, star_screenCnt = contour_detection(edge_frame, frame)
cv2.waitKey(0)
if ball_screenCnt is not None:
    mask_frame = mask(contour_frame, ball_screenCnt)
    text = extract_text(mask_frame)
    print(text)

    cv2.imshow('frame', mask_frame)
    cv2.waitKey(0)
else:
    cv2.imshow('frame', contour_frame)
    cv2.waitKey(0)

'''ball_found = False
# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Edge detection
        edge_frame = edge_detection(frame)
        # Contour detection
        contour_frame, ball_screenCnt, star_screenCnt = contour_detection(edge_frame, frame)
        if ball_screenCnt is not None:
            mask_frame = mask(contour_frame, ball_screenCnt)
            cv2.imshow('frame', contour_frame)
            ball_found = True
        elif ball_found:
            cv2.imshow('frame', contour_frame)
        else:
            cv2.imshow('frame', contour_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break'''


cap.release()
cv2.destroyAllWindows()



