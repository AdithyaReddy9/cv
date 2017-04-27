import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb
from PIL import Image 
import csv
#import mpldatacursor

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


with open('jpegs/utm.csv', 'rb') as f:
    reader = csv.reader(f)

img1 = cv2.imread('jpegs/611.jpeg')
img2 = cv2.imread('jpegs/613.jpeg')
img2 = cv2.resize(img2, (0,0), fx=0.1, fy=0.1) 


fig, ax = plt.subplots()
im = ax.imshow(img2, interpolation='none')
ax.format_coord = Formatter(im)
plt.show()

# translations fro the GPS - (x,y) coordinates
 
rows,cols = img2.shape[:2]
M = np.float32([[1,0,40],[0,1,40]])
dst = cv2.warpAffine(img2,M,(cols,rows))

plt.imshow(dst) # ,plt.show()
mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
plt.show()

#pdb.set_trace()

#if __name__ == '__main__':




