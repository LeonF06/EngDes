import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()