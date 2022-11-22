import cv2
import imutils
import sys
import numpy as np
from matplotlib import pyplot as plt
from Class import *



# Connect to camera
def connect_camera():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    return cap

# Convert frame to grayscale
def convert_to_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

# Threshold the frame
def threshold_frame(gray, thresh_val):
    ret, thresh = cv2.threshold(gray, thresh_val, 255, 0)
    return thresh

# Blur the frame
def blur_frame(gray, x):
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.GaussianBlur(gray, (x, x), 0)
    return gray

# Canny edge detection
def edge_detection(gray):
    edged = cv2.Canny(gray, 10, 250)
    return edged

# Function to create a circular mask
def mask_ball(gray, center, radius):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    # make the background white
    masked[mask == 0] = 255
    return masked

# Contour detection function
def contour_detection(edge):
    contours = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    print("---The number of contours detected: ", len(contours))

    for c in contours:
        contour_area = cv2.contourArea(c)
        approx = calc_approx(c)
        print(" Area: ", contour_area, " Length: ", len(c))

        # Ball is detected
        if (ball.detected == False and contour_area > 140000):
            print("Ball contour")
            ball.detected = True
            #cv2.drawContours(img_copy, [approx], 0, (255, 0, 255), 3)
            return 0, approx, contour_area
        # Ball is not found
        elif (ball.detected == False and contour_area < 140000):
            print("Not ball contour")
            return None, None, None
        # Blob is detected
        elif (blob.detected == False and contour_area > 20000 and contour_area < 65000):
            print("blob contour")
            blob.detected = True
            return 3, approx, contour_area
        # Defect is detected
        elif (contour_area > 5 and contour_area < 20000 and is_closed(c) == True):
            #ellipse = cv2.fitEllipse(c)
            print("defect contour")
            if (intersect(ball.outline, c) == False):
                return 4, approx, contour_area
        else :
            continue
            

# Calculate approx
def calc_approx(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.001*peri, True)
    return approx

# Check if contour is closed
def is_closed(c):
    if (cv2.contourArea(c) > cv2.arcLength(c, True)):
        print("Closed")
        return True
    else:
        print("Open")
        return False

# Function to determine if a point is on the left or right side of a line
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Function to determine if two contours intersect
def intersect(circle_cnt, refernce_cnt):
    for ref_idx in range(len(circle_cnt)-1):
    ## Create reference line_ref with point AB
        A = circle_cnt[ref_idx][0]
        B = circle_cnt[ref_idx+1][0] 
    
        for query_idx in range(len(refernce_cnt)-1):
            ## Create query line_query with point CD
            C = refernce_cnt[query_idx][0]
            D = refernce_cnt[query_idx+1][0]
        
            ## Check if line intersect
            if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
                ## If true, break loop earlier
                return True
    return False

# Crop the image
def crop_frame(gray, approx):
    x, y, w, h = cv2.boundingRect(approx)
    crop_gray = gray[y:y+h+2, x:x+w+2]
    return crop_gray

# Average the grayscale image
def average_frame(gray, size):
    avg = cv2.blur(gray, (size, size))
    return avg

# Function to determine wether the ball is orange or white
def is_orange(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print("Shape of hsv: ", hsv.shape)
    lower_orange = np.array([0, 100, 250])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    if (np.sum(res) > 0):
        print("Orange")
        return True
    else:
        print("White")
        return False


''''''''''''''''''' MAIN FUNCTION '''''''''''''''''''
# Connect to the camera and get a frame
cap = connect_camera()
ball = Ball(None, None, None, None, None, False)
while (ball.detected == False):
    ret, frame = cap.read()

    # Determine wether the ball is orange or white
    ball.colour = is_orange(frame)

    ''''''''''''''''''' BALL DETECTION '''''''''''''''''''
    # Convert the frame to grayscale and then to binary
    gray_img = convert_to_grayscale(frame)
    thresh_img = threshold_frame(gray_img, 130) #190

    # Blur the frame and then perform edge detection
    blur_img = blur_frame(thresh_img, 3)
    edged_img = edge_detection(blur_img)

    # Detect and draw the contours and isolate the ball
    blob = Blob(None, None, False)

    try :
        state, ball_outline, ball_area = contour_detection(edged_img)
        if state == 0 :
            print("Found the ball")

            (x,y),radius = cv2.minEnclosingCircle(ball_outline)
            ball.center = (int(x),int(y))
            ball.radius = int(radius)

            ball.area = ball_area
            #ball_img = crop_frame(gray_img, ball.outline)
            ball_img = cv2.circle(frame, ball.center, ball.radius, (0,255,0), 2)
    except :
        continue

# Get the contour of the ball
ball.detected = False
blank_img = np.zeros(gray_img.shape, np.uint8)
cv2.circle(blank_img, ball.center, ball.radius, (255,255,255), -1)
edged_img = edge_detection(blank_img)
state, ball_outline, ball_area = contour_detection(edged_img)
ball.outline = ball_outline



''''''''''''''''''' BLOB DETECTION '''''''''''''''''''
# Mask the ball image
ball_mask_img = mask_ball(gray_img, ball.center, ball.radius)
ball_img = ball_mask_img.copy()

# Make a white circle around the ball
cv2.circle(ball_img, ball.center, ball.radius-1, (255,255,255), 15)
# Add contrast to the image
ball_img = cv2.addWeighted(ball_img, 2, ball_img, 0, 0)

# Send the contour image through a averaging filter to merge the pixels
if (ball.colour == "White") :
    avg_img = average_frame(ball_img, 20)
    avg_img = cv2.equalizeHist(avg_img)

    for i in range(15):
        avg_img = average_frame(avg_img, 6)
        avg_img = threshold_frame(avg_img, 140) #230
# Orange
else:
    avg_img = average_frame(ball_img, 20)
    avg_img = cv2.equalizeHist(avg_img)

    for i in range(15):
        avg_img = average_frame(avg_img, 7)
        avg_img = threshold_frame(avg_img, 160) #230

binary_img = avg_img.copy()

# Edge detection and contour detection on the binary image
test_img = frame.copy()
cv2.circle(test_img, ball.center, ball.radius, (0,0,0), 1)
edged_img = edge_detection(binary_img)
cv2.circle(binary_img, ball.center, ball.radius, (0,0,0), 1)
try:
    state, blob_outline, blob_area = contour_detection(edged_img)
    if (state == 3):
        print("Found the blob")
        blob.outline = blob_outline
        blob.area = blob_area
        cv2.drawContours(test_img, [blob.outline], 0, (0, 0, 0), 3)
except:
    print("Blob not found")


''''''''''''''''''' DEFECT DETECTION '''''''''''''''''''
# First check if the blob is present in the frame
if (blob.detected == True):
    print("**Im here**")

    try :
        state, defect_outline, defect_area = contour_detection(edged_img)
        if state == 4 :
            print("Found a defect")
            defect = Defect(defect_outline, defect_area)
            #test_img = ball_img.copy()
            cv2.drawContours(test_img, [defect.outline], 0, (255, 0, 255), 3)
    except :
        print("No defects found")
    

# If the blob is not present, then check for defects without blur and merge
else:
    #defect_binary_img = threshold_frame(ball_img, 190)
    #edged_img = edge_detection(defect_binary_img)
    try :
        state, defect_outline, defect_area = contour_detection(edged_img)
        if state == 4 :
            print("Found a defect")
            defect = Defect(defect_outline, defect_area)
            #test_img = ball_img.copy()
            cv2.drawContours(test_img, [defect.outline], 0, (255, 0, 255), 3)
    except :
        print("No defects found")




cv2.imshow('frame', binary_img)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()