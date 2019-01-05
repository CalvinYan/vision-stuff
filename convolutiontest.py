import cv2
import numpy as np

img = cv2.imread("Images/portrait.jpg", cv2.IMREAD_COLOR)

kernel_sharpen = np.array([[0, 0, 0, 0, 0],
                           [0, 0, -1, 0, 0],
                           [0, -1, 5, -1, 0],
                           [0, 0, -1, 0, 0],
                           [0, 0, 0, 0, 0]])

kernel_blur = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0]])

kernel_edge_enhance = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, -1, 1, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])

kernel_edge_detect = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 1, -4, 1, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0]])

kernel_emboss = np.array([[0, 0, 0, 0, 0],
                        [0, -2, -1, 1, 0],
                        [0, -1, 1, 1, 0],
                        [0, 1, 1, 2, 0],
                        [0, 0, 0, 0, 0]])

kernel_custom = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 2, 0, 0],
                        [0, -1, -1, -1, 0],
                        [0, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0]])

# To change the filter, change this line! Don't touch anything else!
kernel = kernel_emboss

print "Convolutional filter used:"
print kernel

new_img = cv2.filter2D(img, -1, kernel)

cv2.imshow("Original image", img)
cv2.imshow("What in convolution", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
