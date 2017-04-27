import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb
import funct

RESCALE = 0.5


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it


img = cv2.imread('jpegs/604.jpeg')
old_frame = cv2.resize(img, (0,0), fx=RESCALE, fy=RESCALE) 
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

i= 604;

images = []
origins =[]
while (1):
    strName = 'jpegs/' + str(i) + '.jpeg'
    img = cv2.imread(strName)
    frame = cv2.resize(img, (0,0), fx=RESCALE, fy=RESCALE) 
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
       
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    M, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0) 
   # M, mask = cv2.findHomography(good_new, good_old, cv2.RANSAC, 5.0) 
    warped_left, origin1 = funct.warp_image(frame, M)
    st = "warp" + str(i)+ ".png"
    cv2.imwrite(st, warped_left)
    images.append(warped_left)
    origins.append(origin1)
    print M
    #pdb.set_trace()
    if i == 607:
	break;
    i = i + 1 


images.append(old_frame);
origins.append([0, 0])
cv2.imwrite("old_frame.png", old_frame)
cv2.imwrite("warped.png", warped_left)
cv2.imwrite("actual.png", frame)
#images = (warped_left, frame)
#origins = (origin1, (0, 0))
mosaic1 = funct.create_mosaic(images, origins)

cv2.imwrite("feetMosaic.png", mosaic1)


