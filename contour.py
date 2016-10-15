import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lowrange = np.array([40,70,130], np.uint8)
highrange = np.array([70,200, 255], np.uint8)

while True:
    ret, img = cap.read()
##    img = cv2.imread('Images/opencv-logo.png')
    img = cv2.resize(img, (320, 240))
    binary = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lowrange, highrange)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imshow("contour-binary", binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 2)
    cv2.imshow("contour", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(0)
cv2.destroyAllWindows()
