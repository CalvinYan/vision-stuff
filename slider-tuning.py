import cv2
import numpy as np
import urllib2

slider_name = 'Adjust HSV threshold'

# Receive the min and max HSV values and filter the image accordingly
def updateThreshold():
    low, high = getTrackbars()
    # Create a binary image using the lower and upper bounds as thresholds
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), low, high)
    hsv = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('Filtered image', hsv)

# Initialize the trackbars that adjust the HSV threshold
def createTrackbars():
    names = np.array(['H low', 'S low', 'V low', 'H high', 'S high', 'V high'])
    values = np.array([[0, 180], [0, 255], [0, 255], [180, 180], [255, 255], [255, 255]])
    for i,v in enumerate(names):
        cv2.createTrackbar(v, slider_name, values[i, 0], values[i, 1], lambda x: None)

# Receive trackbar values in the form of a lower bound and upper bound
def getTrackbars():
    retval = []
    names = np.array(['H low', 'S low', 'V low', 'H high', 'S high', 'V high'])
    for i in names:
        retval.append(cv2.getTrackbarPos(i, slider_name))
    # Return the first three values as the lower bound and the last three as the
    # upper bound
    return np.array(retval[0:3]), np.array(retval[3:6])

# Get the HSV value of a pixel when clicked
def getHSV(event, x,  y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print cv2.cvtColor(img[y:y+1, x:x+1, :], cv2.COLOR_BGR2HSV)

cv2.namedWindow('Adjust HSV threshold', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original image')

# The original image calls getHSV() when clicked
cv2.setMouseCallback('Original image', getHSV)

createTrackbars()

# Connect to the camera
stream = urllib2.urlopen("http://192.168.2.101:8080/videofeed")
bytes=''
paused = False
while True:

    # Read from the camera feed. \xff\xd8 marks the beginning of an image in the
    # byte stream and \xff\xd9 represents the end.
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    
    if a!=-1 and b!=-1:

        # Get raw image data and empty the bytes array
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        # Translate the image into a readable format and resize it
        img = cv2.imdecode(np.fromstring(jpg, np.uint8), 1)
        img = cv2.resize(img, (640, 480))

        # Display the camera feed if the program is not paused and filter it
        if not paused:
            cv2.imshow('Original image', img)
            updateThreshold()

        # Press 'q' to exit the program and 'p' to pause and unpause
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
cv2.destroyAllWindows()
        

