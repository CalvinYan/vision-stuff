import cv2
import numpy as np
import urllib2

cap = cv2.VideoCapture(0)

lowrange = np.array([10,180,130], np.uint8)
highrange = np.array([30,255,255], np.uint8)

stream = urllib2.urlopen("http://192.168.2.105:8080/videofeed")
bytes=''
while True:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, np.uint8), 1)
##    ret, img = cap.read()
##    img = cv2.imread('Images/opencv-logo.png')
        img = cv2.resize(img, (640, 480))
        binary = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lowrange, highrange)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cv2.imshow("contour-binary", binary)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(img, contours, 0, (0, 255, 0), 2)
        cv2.imshow("contour", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
cv2.destroyAllWindows()
