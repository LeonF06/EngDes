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
        if (ball.detected == False and contour_area > 20000):
            print("Ball contour")
            ball.detected = True
            #cv2.drawContours(img_copy, [approx], 0, (255, 0, 255), 3)
            return 0, approx, contour_area
        # Ball is not found
        elif (ball.detected == False and contour_area < 20000):
            print("Not ball contour")
            return None, None, None
        # Inner circle is detected
        elif (inner_circle.detected == False and contour_area > 17000 and contour_area < 20000 and is_closed(c) == True):
            print("inner circle contour")
            inner_circle.detected = True
            return 1, approx, contour_area
        # Small circle is detected
        elif (small_circle.detected == False and contour_area > 14000 and contour_area < 17000 and is_closed(c) == True):
            print("small circle contour")
            small_circle.detected = True
            return 2, approx, contour_area
        # Blob is detected
        elif (blob.detected == False and contour_area > 3500 and contour_area < 10000 and is_closed(c) == True):
            print("blob contour")
            blob.detected = True
            return 3, approx, contour_area
        # Defect is detected
        elif (contour_area > 5 and contour_area < 3500 and is_closed(c) == True):
            #ellipse = cv2.fitEllipse(c)
            print("defect contour")
            if (intersect(c, small_circle.outline) == False):
                return 4, approx, contour_area
        else :
            print("No contour")
            continue
            

# Calculate approx
def calc_approx(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.001*peri, True)
    return approx

# Check if contour is closed
def is_closed(c):
    if (cv2.contourArea(c) > cv2.arcLength(c, True)):
        return True
    else:
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



''''''''''''''''''''''''' MAIN FUNCTION '''''''''''''''''''''''''
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        ''''''''''''''''''' BALL DETECTION '''''''''''''''''''
        # Convert the frame to grayscale and then to binary
        gray_img = convert_to_grayscale(frame)
        thresh_img = threshold_frame(gray_img, 180) #190

        # Blur the frame and then perform edge detection
        blur_img = blur_frame(thresh_img, 3)
        edged_img = edge_detection(blur_img)

        # Detect and draw the contours and isolate the ball
        ball = Ball(None, None, False)
        blob = Blob(None, None, False)

        try :
            state, ball_outline, ball_area = contour_detection(edged_img)
            if state == 0 :
                print("Found the ball")
                ball.outline = ball_outline
                ball.area = ball_area
                ball_img = crop_frame(gray_img, ball.outline)
        except :
            print("Ball not found")
            exit()


        ''''''''''''''''''' INNER CIRCLE DETECTION '''''''''''''''''''
        # Find the center of the cropped ball image
        cX = ball_img.shape[0] // 2
        cY = ball_img.shape[1] // 2

        # Create a white circle with black background
        circle_img1 = np.zeros((ball_img.shape[0], ball_img.shape[1]), np.uint8)
        cv2.circle(circle_img1, (cX, cY), cX-10, (255, 255, 255), 12)
        # Create a circle image with thinner outline
        circle_img2 = np.zeros((ball_img.shape[0], ball_img.shape[1]), np.uint8)
        cv2.circle(circle_img2, (cX, cY), cX-10, (255, 255, 255), 1)

        # Detect the circle countour
        inner_circle = InnerCircle(None, None, False)
        state, circle_outline, circle_area = contour_detection(circle_img2)
        if state == 1:
            print("Found the inner circle")
            inner_circle.outline = circle_outline
            inner_circle.area = circle_area
            test_img = ball_img.copy()
            cv2.drawContours(test_img, [inner_circle.outline], 0, (0, 0, 0), 3)

        # Detect the small circle countour
        small_circle = SmallCircle(None, None, False)
        state, circle_outline, circle_area = contour_detection(circle_img1)
        if state == 2:
            print("Found the small circle")
            small_circle.outline = circle_outline
            small_circle.area = circle_area
            #test_img = ball_img.copy()
            cv2.drawContours(test_img, [small_circle.outline], 0, (0, 0, 0), 3)


        ''''''''''''''''''' BLOB DETECTION '''''''''''''''''''
        # Send the contour image through a averaging filter to merge the pixels
        avg_img = average_frame(ball_img, 10)
        for i in range(15):
            avg_img = average_frame(avg_img, 5)

        # Convert the blurry frame to bianry
        binary_img = threshold_frame(avg_img, 220) #230

        # Combine the circle image and the binary image
        combined_img = cv2.bitwise_or(binary_img, circle_img1)

        # Edge detection and contour detection on the binary image
        edged_img = edge_detection(combined_img)
        try:
            state, blob_outline, blob_area = contour_detection(edged_img)
            if (state == 3):
                print("Found the blob")
                blob.outline = blob_outline
                blob.area = blob_area
                #test_img = ball_img.copy()
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
                    test_img = ball_img.copy()
                    cv2.drawContours(test_img, [defect.outline], 0, (0, 0, 0), 3)
                    #cv2.ellipse(test_img, defect.outline, (0, 0, 0), 3)
            except :
                print("No defects found")
            

        # If the blob is not present, then check for defects without blur and merge
        else:
            defect_binary_img = threshold_frame(ball_img, 190)
            edged_img = edge_detection(defect_binary_img)
            try :
                state, defect_outline, defect_area = contour_detection(edged_img)
                if state == 4 :
                    print("Found a defect")
                    defect = Defect(defect_outline, defect_area)
                    #test_img = ball_img.copy()
                    cv2.drawContours(test_img, [defect.outline], 0, (0, 0, 0), 3)
            except :
                print("No defects found")
    else:
        break

cap.release()
cv2.destroyAllWindows()