import cv2
import numpy as np

#Samsung Galaxy S5
intrinsic = np.array([[  3952.76282,             0,   2113.31139],
                     [            0,    3846.76551,   1029.12655],
                     [            0,             0,   1]])

warp = np.array([[  2.24863294,    -.154037417,   0],
                 [  .660589253,     1.59248079,   0],
                 [.00223121669, -.000287467308,   1.00000000]])

rotationZ = np.array([[  0.8660254,    0.5,   100],
                     [  -0.5,     0.8660254,   500],
                     [0, 0,   1.00000000]])

rotationY = np.array([[  0.8660254,    0,   0.5],
                     [  0,     1,   0],
                     [-0.5, 0,   0.8660254]])

rotationX = np.array([[  1,    0,   0],
                     [  0,      0.98480775,   0.17364818],
                     [0, -0.17364818,   0.98480775]])

def getIdentity():
    corners = np.array([[0, 0], [639, 0], [0, 383], [639, 383]], dtype=np.float32)
    H, _ = cv2.findHomography(corners, corners, cv2.RANSAC)
    print H

goal = cv2.imread("./Images/Goal.png", cv2.IMREAD_COLOR)
cv2.imshow("Goal", goal)
identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
transform = np.divide(identity, intrinsic)
M = np.multiply(intrinsic, transform)
print cv2.getRotationMatrix2D((320, 192), 10, 1.0)
cv2.imshow("Should look the same", cv2.warpPerspective(goal, rotationY, (1280, 960)))
getIdentity()
cv2.waitKey(0)
cv2.destroyAllWindows()
