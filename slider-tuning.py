import cv2
import numpy as np
import urllib2


slider_name = 'Adjust HSV threshold'


def nothing(x):
    hLow, sLow, vLow, hHigh, sHigh, vHigh = getTrackbars()
    lower_range = np.array([hLow, sLow, vLow], np.uint8)
    higher_range = np.array([hHigh, sHigh, vHigh], np.uint8)
    hsv = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_range, higher_range);
    cv2.imshow("Filtered", hsv)

def createTrackbars():
    cv2.createTrackbar('H low', slider_name, 0, 180, nothing)
    cv2.createTrackbar('S low', slider_name, 0, 255, nothing)
    cv2.createTrackbar('V low', slider_name, 0, 255, nothing)
    cv2.createTrackbar('H high', slider_name, 180, 180, nothing)
    cv2.createTrackbar('S high', slider_name, 255, 255, nothing)
    cv2.createTrackbar('V high', slider_name, 255, 255, nothing)

def getTrackbars():
    retval = []
    bar_names = ['H low', 'S low', 'V low', 'H high', 'S high', 'V high']
    for i in range(0, 6):
        retval.append(cv2.getTrackbarPos(bar_names[i], slider_name))
    return retval

def getHSV(event, x,  y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print cv2.cvtColor(img[y:y+1, x:x+1, :], cv2.COLOR_BGR2HSV)
        
##cap = cv2.VideoCapture(0)

cv2.namedWindow('Adjust HSV threshold', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original image')

cv2.setMouseCallback('Original image', getHSV)

createTrackbars()

stream = urllib2.urlopen("http://192.168.2.101:8080/videofeed")
bytes=''
##while cap.isOpened():
while True:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, np.uint8), 1)
        hLow, sLow, vLow, hHigh, sHigh, vHigh = getTrackbars()
        lower_range = np.array([hLow, sLow, vLow], np.uint8)
        higher_range = np.array([hHigh, sHigh, vHigh], np.uint8)
    ##    ret, img = cap.read()
    ##    img = cv2.imread('Images/opencv-logo.png')
        img = cv2.resize(img, (640, 480))
        kernel = np.ones((5,5),np.uint8)
        hsv = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_range, higher_range);
        mod = cv2.morphologyEx(cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
        cv2.imshow('Original image', img)
        cv2.imshow('Filtered', hsv)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
cv2.destroyAllWindows()
