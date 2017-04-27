import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb

RESCALE = 0.5
def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """

    p1 = np.ones(3, np.float32)
    p2 = np.ones(3, np.float32)
    p3 = np.ones(3, np.float32)
    p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    p1[:2] = [0, 0]
    p2[:2] = [x, 0]
    p3[:2] = [0, y]
    p4[:2] = [x, y]

    min_x = None
    min_y = None
    max_x = None
    max_y = None

    for pt in [p1, p2, p3, p4]:
        hp = np.dot(np.matrix(homography, np.float32),
                    np.matrix(pt, np.float32).T)

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([[hp_arr[0] / hp_arr[2]],
                             hp_arr[1] / hp_arr[2]], np.float32)

        if(max_x is None or normal_pt[0, 0] > max_x):
            max_x = normal_pt[0, 0]

        if(max_y is None or normal_pt[1, 0] > max_y):
            max_y = normal_pt[1, 0]

        if(min_x is None or normal_pt[0, 0] < min_x):
            min_x = normal_pt[0, 0]

        if(min_y is None or normal_pt[1, 0] < min_y):
            min_y = normal_pt[1, 0]
       
        print min_x, min_y, max_x, max_y
    translationMatrix = np.zeros(shape=(3, 3))
    translationMatrix[0] = [0, 0, -min_x]
    translationMatrix[1] = [0, 0, -min_y]
    newHomography = np.add(homography, translationMatrix)
    
    warp = cv2.warpPerspective(image,
                               newHomography,
                               (int(max_x - min_x), int(max_y - min_y)))

   # pdb.set_trace()
    return warp, (int(min_x), int(min_y))


def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.

    Arguments:
      images: a list of 4-channel images to combine in the mosaic.
      origins: a list of the locations upper-left corner of each image in
               a common frame, e.g. the frame of a central image.

    Returns: a new 4-channel mosaic combining all of the input images. pixels
             in the mosaic not covered by any input image should have their
             alpha channel set to zero.
    """
    # This will make the first image, in the images list,
    # have an origin of (0,0)
    # so that we can stitch them sequentially.
    new_origins = []
    o_x, o_y = origins[0]
    for origin in origins:
        x, y = origin
        new_origins.append([x + abs(o_x), y + abs(o_y)])

    # Dimensions for the mosaic
    max_height = 0
    max_width = 0
    for image, origin in zip(images, new_origins):
        x, y = image.shape[:2]
        max_width += origin[0]
        max_height += origin[1]

    max_width += max(images[0].shape[1],images[1].shape[1]) 
    max_height += max(images[0].shape[0], images[1].shape[0])

    final = np.ones((max_height, max_width, images[0].shape[2]), np.uint8)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2BGRA)
   # pdb.set_trace()
    for i in range(len(images)):
        y, x, _ = images[i].shape
        o_x, o_y = new_origins[i]

        final[abs(o_y):abs(y) + abs(o_y),
              abs(o_x):abs(x) + abs(o_x), :_] = images[i]

    # cv2.imshow('final', final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return final



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
img = cv2.imread('jpegs/831.jpeg')
old_frame = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

mask = np.zeros_like(old_frame)

img = cv2.imread('jpegs/832.jpeg')
frame = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
   
# Select good points
good_new = p1[st==1]
good_old = p0[st==1]
M, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 20.0) 
#M, mask = cv2.findHomography(good_new, good_old, cv2.RANSAC, 5.0) 
warped_left, origin1 = warp_image(old_frame, M)

cv2.imwrite("old_frame.png", old_frame)
cv2.imwrite("old_warped.png", warped_left)
cv2.imwrite("new_frame.png", frame)
images = (warped_left, frame)
origins = (origin1, (0, 0))
mosaic1 = create_mosaic(images, origins)

cv2.imwrite("feetMosaic.png", mosaic1)

pdb.set_trace()


# draw the tracks
for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
img = cv2.add(frame,mask)

cv2.imshow('frame',img)
k = cv2.waitKey(30) & 0xff
    # Now update the previous frame and previous points
old_gray = frame_gray.copy()
p0 = good_new.reshape(-1,1,2)

