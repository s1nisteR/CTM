from imutils.video import VideoStream
import imutils
import time
import cv2
import math
import numpy as np

'''
Instructions to use:
https://chev.me/arucogen/

Dictionary: 5x5
Marker ID: any number
Marker size 100mm
Scan the generated marker with video source 
'''

'''
Ideal settings for an ArUco dictionary include::
1. A low number of unique ArUco IDs that need to be generated and read
2. High-quality image input containing the ArUco markers that will be detected
3. A larger NxN grid size, balanced with a low number of unique ArUco IDs such
that the inter-marker distance can be used to correct misread markers
'''

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_100"])
arucoParams = cv2.aruco.DetectorParameters_create()

locations = ""

vs = VideoStream(src=0).start() #change to required video source
# time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    aruco_points = []
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    height, width, channels = frame.shape
    
    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

    # verify *at least* one ArUco marker was detected
    
    
    
    if len(corners) > 0:        
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            
            aruco_points.append((cX, cY))
            
            cv2.putText(frame, str(markerID) + f" {str(cX)} | {str(cY)}", (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 100), 2)

    cv2.putText(frame, str(width)  + " WIDTH", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, str(height) + " HEIGHT", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    locations = str(aruco_points)
    cv2.putText(frame, locations, (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) 
    
    # Center of the frame cube
    cv2.rectangle(frame, (int(width/2)-5, int(height/2)-5), (int(width/2)+5, int(height/2)+5), (0, 0, 255), 2)
    
    


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

