import cv2
import numpy as np
import math

# Image width, image height, and camera focal distance
w = 640
h = 384
f = 620

# Given six parameters: the rotations aroung the x, y, and z axes, as well as the
# translations along those axes, compute the tranformation matrix and warp the image
def warp(angles, translations):
    angles = [math.radians(x) for x in angles]
    S = [math.sin(x) for x in angles]
    C = [math.cos(x) for x in angles]

    print angles

    # The rotation matrices along the x, y, and z axes
    X = np.array([[     1,        0,       0, 0],
                  [     0,     C[0],    S[0], 0],
                  [     0,    -S[0],    C[0], 0],
                  [     0,        0,       0, 1]], np.float32)

    Y = np.array([[  C[1],     0,   S[1], 0],
                         [   0,     1,   0, 0],
                         [  -S[1],     0,   C[1], 0],
                  [0, 0, 0, 1]], np.float32)

    Z = np.array([[  C[2],     S[2],   0, 0],
                  [ -S[2],     C[2],   0, 0],
                  [     0,        0,   1, 0],
                  [     0,        0,   0, 1]], np.float32)

    # Compute the 3d rotation matrix using X * Y * Z
    R = np.dot(np.dot(X, Y), Z)

    # 3D translation matrix
    T = np.array([[   1,     0,   0, translations[0]],
                  [   0,     1,   0, translations[1]],
                  [   0,     0,   1, translations[2]],
                  [   0,     0,   0,               1]], np.float32)

    # Intrinsic camera parameters; used to convert from 2d data to 3d and back to 2d
    A1 = np.array([[1, 0, -w/2],
                   [0, 1, -h/2],
                   [0, 0,    0],
                   [0, 0,    1]])
    A2 = np.array([[f, 0, -w/2, 0],
                   [0, f, -h/2, 0],
                   [0, 0,    1, 0]])

    # Perspective transformation matrix = A2 * (T * (R * A1))
    M = np.dot(A2, np.dot(T, np.dot(R, A1)))

    # Normalize the matrix so the bottom right value is 1
    M = np.divide(M, M[2, 2])
##    print M
    findPose(M)
    return cv2.warpPerspective(goal, M, (1280, 960))

def zoom(x):
    cv2.imshow("Should look the same", warp([x, 0, 0], [1000, 600, 1000]))

def findPose(M):
    col1 = np.array(M[:,0])
    col2 = np.array(M[:,1])
    col3 = np.cross(col1, col2)
    col4 = np.array(M[:, 2])
    projection = np.column_stack((col1, col2, col3, col4))
    camera, rotation, translation, rotX, rotY, rotZ, euler = cv2.decomposeProjectionMatrix(projection)
    #print np.divide(camera, camera[2, 2])
    euler = np.array([np.radians(x) for x in euler], np.float32)
    print euler
##    print np.divide(rotX, rotX[0,0])

cv2.namedWindow("Set zoom", 0)
cv2.createTrackbar("Z translation", "Set zoom", 0, 360, zoom)
goal = cv2.imread("./Images/Goal.png", cv2.IMREAD_COLOR)
cv2.imshow("Goal", goal)
cv2.imshow("Should look the same", warp([0, 0, 0], [1000, 600, 1000]))
cv2.waitKey(0)
cv2.destroyAllWindows()
