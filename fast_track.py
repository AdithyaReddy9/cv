import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb

img = cv2.imread('jpegs/650.jpeg', 0)

#print img.shape
#pdb.set_trace()
#gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,None)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
pdb.set_trace()
img2 =  cv2.drawKeypoints(img, kp, None, color=(255,0,0))


# Print all default params
print "Threshold: ", fast.getInt('threshold')
print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
print "neighborhood: ", fast.getInt('type')
print "Total Keypoints with nonmaxSuppression: ", len(kp)

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)

print "Total Keypoints without nonmaxSuppression: ", len(kp)

img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)
