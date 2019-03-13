from q2 import makeTestPattern, computeBrief, briefLite, briefMatch, testMatch,briefRotTest, briefRotLite

import scipy.io as sio
import skimage.color
import skimage.io
import skimage.feature
import skimage.transform
import numpy as np

# Q2.1
compareX, compareY = makeTestPattern(9,256)
sio.savemat('testPattern.mat',{'compareX':compareX,'compareY':compareY})

# Q2.2
img = skimage.io.imread('../data/chickenbroth_01.jpg')
img_compare=skimage.io.imread('../data/model_chickenbroth.jpg')
im = skimage.color.rgb2gray(img)
im_compare=skimage.color.rgb2gray(img_compare)

# YOUR CODE: Run a keypoint detector, with nonmaximum supression
# locs holds those locations n x 2
locs = None
locs = skimage.feature.corner_peaks(skimage.feature.corner_harris(im,1.5), min_distance=1)
locs, desc = computeBrief(im,locs,compareX,compareY)

# Q2.3
locs, desc = briefLite(im)

# Q2.4
testMatch(im_compare,im)

# Q2.5
briefRotTest()

# EC 1
#briefRotTest(briefRotLite)

# EC 2 
# write it yourself!
im1=skimage.transform.resize(im,im.shape*np.asarray([1/2]))
testMatch(im,im1)