import cv2
import numpy as np
import urllib2
from operator import itemgetter

cap = cv2.VideoCapture(0)

lowrange = np.array([18,160,130], np.uint8)
highrange = np.array([30,255,200], np.uint8)

stream = urllib2.urlopen("http://192.168.2.101:8080/videofeed")
bytes=''

def findCenter():
    src = np.array([[0, 0], [320, 0], [0, 240], [320, 240]], dtype=np.float32)
    dst = []
    colors = np.array([(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)])
    if len(contours) > 0:
        contour = contours[0]
        contour = sorted(contour[:, 0, :], key=lambda x: x[0] + x[1], reverse=False)
        topLeft = [contour[0][0], contour[0][1]]
        bottomRight = [contour[-1][0], contour[-1][1]]
        contour = sorted(contour, key=lambda x: x[0] - x[1], reverse=False)
        bottomLeft = [contour[0][0], contour[0][1]]
        topRight = [contour[-1][0], contour[-1][1]]
        dst.append(topLeft)
        dst.append(topRight)
        dst.append(bottomLeft)
        dst.append(bottomRight)
        for i in range(0, 4):
            cv2.circle(img, (dst[i][0], dst[i][1]), 5, colors[i], -1)
        dst = np.array(dst, dtype=np.float32)
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        center = cv2.perspectiveTransform(np.array([[[160, 120]]], np.float32), M)
        center = center[0][0]
        cv2.line(img, (320, 240), (center[0], center[1]), (0, 255, 0), 2)
        return dst, center
        

def findDepth():
    width1 = cv2.norm(np.array(dst[1][0] - dst[0][0], dst[1][1] - dst[0][1]))
    width2 = cv2.norm(np.array(dst[3][0] - dst[2][0], dst[3][1] - dst[2][1]))
    width = (width1 + width2) / 2
    print width
    if width > 0:
        dist = 20 * 585 / width
        print dist
    

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
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
##        cv2.drawContours(img, contours, 0, (0, 255, 0), 2)
        if len(contours) > 0: 
            dst, center = findCenter()
            findDepth()
        cv2.imshow("contour", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
cv2.destroyAllWindows()
