from q2 import briefLite, briefMatch
from q3 import computeH, computeHnorm, computeHransac, compositeH, HarryPotterize

import skimage.color
import skimage.io
import numpy as np
import math
# make a test case!
# you should write your own
# Create H, x1 and x2 whose answer you know
# Make sure you can recover it!
H, x1, x2 = None, None, None
img = skimage.io.imread('../data/chickenbroth_01.jpg')
#img=skimage.io.imread('../data/cv_cover.jpg')
img_compare=skimage.io.imread('../data/model_chickenbroth.jpg')
im = skimage.color.rgb2gray(img)
im_compare=skimage.color.rgb2gray(img_compare)
locs1,desc1=briefLite(im)
locs2,desc2=briefLite(im_compare)
match=briefMatch(desc1,desc2,ratio=0.8)
x1=locs1[match[:,0],:]
x2=locs2[match[:,1],:]
# 3.1
H2to1 = computeH(x1, x1)
H2to1 = H2to1/H2to1[2,2]
print('should be identity\n',H2to1,'\n')

H2to1 = computeH(x1, x2)
H2to1 = H2to1/H2to1[2,2]
print('normal\n',H2to1,'\n')

# 3.2
H2to1 = computeHnorm(x1, x2)
H2to1 = H2to1/H2to1[2,2]
print('normalized\n',H2to1,'\n')

# 3.3 
bestH2to1, inliers = computeHransac(x1,x2)
bestH2to1 = bestH2to1/bestH2to1[2,2]
print('ransac\n',bestH2to1,'\n')

# 3.4
HarryPotterize()