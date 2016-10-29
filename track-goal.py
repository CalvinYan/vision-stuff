import cv2
import numpy as np
import urllib2
import math

cap = cv2.VideoCapture(0)

lowrange = np.array([10,150,100], np.uint8)
highrange = np.array([30,255,255], np.uint8)

stream = urllib2.urlopen("http://192.168.2.101:8080/videofeed")
bytes=''

#Samsung Galaxy S5
intrinsic = np.array([[  3956.81689,             0,   2383.97949],
                     [            0,    3891.87826,   1205.53979],
                     [            0,             0,   1]])

def findCenter():
    src = np.array([[0, 0], [639, 0], [0, 383], [639, 383]], dtype=np.float32)
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
        dst = np.array([topLeft, topRight, bottomLeft, bottomRight], np.float32)
        for i in range(0, 4):
            cv2.circle(img, (dst[i][0], dst[i][1]), 5, colors[i], -1)
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC)
##        transform = np.multiply(intrinsic, np.linalg.inv(intrinsic))
        transform = np.divide(M, intrinsic)
        print M
        print transform
##        print cv2.getRotationMatrix2D((160, 120), 10, 1.0)
##        print M
##        print np.multiply(transform, intrinsic)
##        center = np.multiply(np.multiply(transform, intrinsic), np.array([160, 120, 1], np.float32))
        center = cv2.perspectiveTransform(np.array([[[320, 192]]], np.float32), M)
##        print center
        M, _ = cv2.findHomography(dst, src, cv2.RANSAC)
        M[0:2, 2:3] = np.array([[0], [0]], np.float32)
##        intrinsic[0:2, 2:3] = np.zeros((2, 1))
        center = center[0][0]
        cv2.line(img, (320, 240), (center[0], center[1]), (0, 255, 0), 2)
        return dst, center
        

def findDepth():
    x, y, w, h = cv2.boundingRect(contours[0])
##    print "Width: %s" % w
    if w > 0:
        zDist = (20 * 620) / w
        print "Distance to goal: %s" % zDist
        xDist = (center[0] - 320) / w * 20
        print "Distance to center: %s" % xDist
        azimuth = math.degrees(math.atan(xDist / zDist))
        print "Azimuth angle: %s" % azimuth
        cv2.putText(img, "%s" % azimuth, (0, 479), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    

while True:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, np.uint8), 1)
        img = cv2.resize(img, (640, 480))
        binary = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lowrange, highrange)
        cv2.imshow("Filtered", binary)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(img, contours, 0, (0, 255, 0), 2)
        if len(contours) > 0: 
            dst, center = findCenter()
            findDepth()
        cv2.imshow("contour", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
cv2.destroyAllWindows()
