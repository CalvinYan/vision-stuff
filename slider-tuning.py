import cv2
import numpy as np
import urllib

def nothing(x):
    pass

def createTrackbars():
    cv2.createTrackbar('H low', 'image', 0, 180, nothing)
    cv2.createTrackbar('S low', 'image', 0, 255, nothing)
    cv2.createTrackbar('V low', 'image', 0, 255, nothing)
    cv2.createTrackbar('H high', 'image', 180, 180, nothing)
    cv2.createTrackbar('S high', 'image', 255, 255, nothing)
    cv2.createTrackbar('V high', 'image', 255, 255, nothing)
    
cap = cv2.VideoCapture(0)

cv2.namedWindow('image')

#cv2.resize('image', (200, 400))

createTrackbars()

while True:
    hLow = cv2.getTrackbarPos('H low', 'image')
    sLow = cv2.getTrackbarPos('S low', 'image')
    vLow = cv2.getTrackbarPos('V low', 'image')
    hHigh = cv2.getTrackbarPos('H high', 'image')
    sHigh = cv2.getTrackbarPos('S high', 'image')
    vHigh = cv2.getTrackbarPos('V high', 'image')
    lower_range = np.array([hLow, sLow, vLow], np.uint8)
    higher_range = np.array([hHigh, sHigh, vHigh], np.uint8)
    ret, img = cap.read()
    kernel = np.ones((5,5),np.uint8)
    #mod = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel)
    mod = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), lower_range, higher_range);
    cv2.imshow('wtf', img)
    cv2.imshow('kek', mod)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
