import cv2
import numpy as np
import pdb

img = cv2.imread('jpegs/611.jpeg')

img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

#sift = cv2.SIFT()
kp = sift.detect(gray,None)

pdb.set_trace()
img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints.jpg',img)

img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

