import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Images/sudoku.png", cv2.IMREAD_GRAYSCALE)

if img is not None:
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    canny = cv2.Canny(img, 100, 200)
    cv2.imshow("Canny", canny)
    cv2.imshow("Laplacian", laplacian)
    cv2.imshow("Sobel X", sobelx)
    cv2.imshow("Sobel Y", sobely)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print "Well shit"
