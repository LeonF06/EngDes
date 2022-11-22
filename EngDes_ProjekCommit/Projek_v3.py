import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from Class import *
import time
import serial
from tkinter import *
from PIL import Image, ImageTk
from pprint import pprint


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
        elif (blob.detected == False and contour_area > 20000 and contour_area < 65000 and is_closed(c) == True):
            print("blob contour")
            blob.detected = True
            return 3, approx, contour_area
        # Defect is detected
        elif (contour_area > 2000 and contour_area < 20000 and is_closed(c) == True):
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

# Function to determine wether the ball is orange or white
def is_orange(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 100, 250])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    if (np.sum(res) > 0):
        return True
    else:
        return False


# Function to detect the ball
def detect_ball(frame, run) :

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
    try :
        state, ball_outline, ball_area = contour_detection(edged_img)
        if state == 0 :
            print("Found the ball")

            (x,y),radius = cv2.minEnclosingCircle(ball_outline)
            ball.center = (int(x),int(y))
            ball.radius = int(radius)

            ball.area = ball_area
            ball_img = cv2.circle(frame, ball.center, ball.radius, (0,255,0), 2)
            return ball_img
        else :
            return None
    except :
        return None



def analyze_frame(ball_img, ball, blob):

    # Get the contour of the ball
    ball.detected = False
    gray_img = convert_to_grayscale(ball_img)
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
    cv2.circle(ball_img, ball.center, ball.radius-1, (255,255,255), 10)
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
    d = 0
    if (blob.detected == True):
        print("**Im here**")

        try :
            state, defect_outline, defect_area = contour_detection(edged_img)
            if state == 4 :
                print("Found a defect")
                d = 1
                defect = Defect(defect_outline, defect_area)
                cv2.drawContours(test_img, [defect.outline], 0, (255, 0, 255), 3)
        except :
            print("No defects found")
        

    # If the blob is not present, then check for defects without blur and merge
    else:
        try :
            state, defect_outline, defect_area = contour_detection(edged_img)
            if state == 4 :
                print("Found a defect")
                d = 1
                defect = Defect(defect_outline, defect_area)
                cv2.drawContours(test_img, [defect.outline], 0, (255, 0, 255), 3)
        except :
            print("No defects found")

    return test_img, d

''''''''''''''''''' GUI FUNCTIONs '''''''''''''''''''
def stop():
    global running
    running = False

    serialcomm.write("def".encode('utf-8'))
    amber_Label.place_forget()
    green_Label.place(x=230,y=50)

    pprint(output)

    e.delete(0, END)
    for x in output:
        #e.insert(0, "Output Queue: ")
        e.insert(END,x)

def start():
    global begin
    begin = True
    global running
    running = True

def lights():
    serialcomm.write("lights".encode('utf-8'))
    
''''''''''''''''''' MAIN FUNCTION '''''''''''''''''''
# Connect to the camera and get a frame
ball = Ball(None, None, None, None, None, False)
blob = Blob(None, None, False)

cap = connect_camera()
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

serialcomm = serial.Serial('COM3', 9600)
run = True
data = '0'
serialcomm.timeout = 0.5
count = 0
seconds = 15

global begin
begin = False
global running
running = False

root = Tk()
root.title("Optical Inspection")
root.geometry('640x800')

canvas = Canvas(root, width=cap_width, height=cap_height)
canvas.place(x=0,y=300)

green = PhotoImage(file='EngDes_ProjekCommit\light_green.png')
green_Label = Label(image=green)
green_Label.place(x=230,y=50)

amber = PhotoImage(file='EngDes_ProjekCommit\light_amber.png')
amber_Label = Label(image=amber)
red = PhotoImage(file='EngDes_ProjekCommit\light_red.png')
red_Label = Label(image=red)

button_stop = Button(root, text="Stop", command=stop, bg="red",padx=40,pady=5)
button_stop.place(x=530,y=0)

button_start = Button(root, text="Start", command=start, bg="green",padx=40,pady=5)
button_start.place(x=0,y=0)

button_lights = Button(root, text="Lights", command=lights, bg="blue",padx=40,pady=5)
button_lights.place(x=260,y=0)

e = Entry(root, width=50)
e.place(x=170,y=260)

root.update_idletasks()
root.update()

output = []

'''if cap.isOpened():
    global running
    running = True'''

while begin == False:
    ret, frame = cap.read()
    frame_copy =  frame.copy()

    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image = Image.fromarray(frame_copy))
    canvas.create_image(0, 0, image = photo, anchor=NW)
    root.update_idletasks()
    root.update()

while cap.isOpened() and begin == True:
    ret, frame = cap.read()
    frame_copy =  frame.copy()

    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image = Image.fromarray(frame_copy))
    canvas.create_image(0, 0, image = photo, anchor=NW)
    root.update_idletasks()
    root.update()

    if ret:
        ball = Ball(None, None, None, None, None, False)
        blob = Blob(None, None, False)
        count = count + 1
        print("---------------------------------->>> Count = ", count)
        e.delete(0, END)
        e.insert(0, "Analysing.")
        ball_img = detect_ball(frame, run)
        if (ball_img is not None):
            test_img, defect = analyze_frame(ball_img, ball, blob)
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            if (run == True and defect == 0) :
                green_Label.place_forget()
                amber_Label.place(x=230,y=50)
                serialcomm.write("run".encode('utf-8'))
                print("--------------------Serial write 'run' sent--------------------")
                run = False
            photo = ImageTk.PhotoImage(image = Image.fromarray(test_img))
            canvas.create_image(0, 0, image = photo, anchor=NW)
        else :
            photo = ImageTk.PhotoImage(image = Image.fromarray(frame_copy))
            canvas.create_image(0, 0, image = photo, anchor=NW)
            continue
        root.update_idletasks()
        root.update()

        print("defect: ", defect)
        if defect == 1:
            # add a 1 delay
            end_time = time.time() + 0.5
            while time.time() < end_time:
                root.update_idletasks()
                root.update()
            end_time = 0

            serialcomm.write("def".encode('utf-8'))
            print("--------------------Serial write 'def' sent---------------------")
            #time.sleep(0.5) 
            while (data  != 'end'):
                try : 
                    data = serialcomm.readline().decode('ascii')
                except:
                    continue
            
            print("----------------Data defect------------------", data)
            
            run = True
            print("Place new ball on roller...")
            data = '0'
            output.append(1)

            e.delete(0, END)
            e.insert(0, "Found Defect. Place new ball on rollers.")

            amber_Label.place_forget()
            green_Label.place(x=230,y=50)
            photo = ImageTk.PhotoImage(image = Image.fromarray(test_img))
            canvas.create_image(0, 0, image = photo, anchor=NW)
            #root.update_idletasks()
            #root.update()
            #cv2.VideoCapture.clear(cap)
            cv2.VideoCapture.release(cap)
            frame = None
            cap = connect_camera()
            ret, frame = cap.read()
            running = True

            end_time = time.time() + seconds
            while time.time() < end_time:
                '''ret, frame = cap.read()
                frame_copy =  frame.copy()
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image = Image.fromarray(frame_copy))
                canvas.create_image(0, 0, image = photo, anchor=NW)'''
                #ret, frame = cap.read()
                root.update_idletasks()
                root.update()

                if running == False:
                    break
            end_time = 0

            #time.sleep(15)
        elif defect == 0:
            # try to search for end statement
            try:
                data = serialcomm.readline().decode('ascii')
                print("----------------Data no defect------------------", data)
                if data == 'end':
                    run = True
                    print("Place new ball on roller...")
                    data = '0'
                    output.append(0)

                    e.delete(0, END)
                    e.insert(0, "No Defect. Place new ball on rollers.")

                    amber_Label.place_forget()
                    green_Label.place(x=230,y=50)
                    #photo = ImageTk.PhotoImage(image = Image.fromarray(frame_copy))
                    #canvas.create_image(0, 0, image = photo, anchor=NW)
                    #root.update_idletasks()
                    #root.update()

                    end_time = time.time() + seconds
                    while time.time() < end_time:
                        ret, frame = cap.read()
                        frame_copy =  frame.copy()
                        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                        photo = ImageTk.PhotoImage(image = Image.fromarray(frame_copy))
                        canvas.create_image(0, 0, image = photo, anchor=NW)
                        root.update_idletasks()
                        root.update()

                        if running == False:
                            break
                    end_time = 0
                    #time.sleep(15)
            except:
                continue
        # Add a 100ms delay
        '''end_time = time.time() + 0.5
        while time.time() < end_time:
            root.update_idletasks()
            root.update()
        end_time = 0'''
    else:
        break

    while running == False:
        data = '0'
        ret, frame = cap.read()
        frame_copy =  frame.copy()

        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image = Image.fromarray(frame_copy))
        canvas.create_image(0, 0, image = photo, anchor=NW)
        root.update_idletasks()
        root.update()



'''serialcomm.write("def".encode('utf-8'))
amber_Label.place_forget()
green_Label.place(x=230,y=50)

pprint(output)

e.delete(0, END)
for x in output:
    e.insert(0, "Output Queue: ")
    e.insert(END,x + " ")'''

#cap.release()
#serialcomm.close()

'''while True:
    root.update_idletasks()
    root.update()'''
