from q4 import *
from q3 import *
from q2 import *
import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import skimage.io

# Q 4.1
# load images into img1
# and img2
# compute H2to1
# please use any feature method you like that gives
# good results
img1 = skimage.io.imread('../data/pnc1.png')
img2 = skimage.io.imread('../data/pnc0.png')
bestH2to1 = None
# YOUR CODE HERE
img1_gray=skimage.color.rgb2gray(img1)
img2_gray=skimage.color.rgb2gray(img2)
locs1,desc1=briefLite(img1_gray)
locs2,desc2=briefLite(img2_gray)
match=briefMatch(desc1,desc2,ratio=0.8)
x1=locs1[match[:,0],:]
x2=locs2[match[:,1],:]
#match=np.transpose(pts)
x1[:,[0,1]] = x1[:,[1,0]]
x2[:,[0,1]] = x2[:,[1,0]]
bestH2to1,inliers=computeHransac(x1,x2)
panoImage = imageStitching(img1,img2,bestH2to1)
plt.subplot(1,2,1)
plt.imshow(img1)
plt.title('pnc1')
plt.subplot(1,2,2)
plt.title('pnc0')
plt.imshow(img2)
plt.figure()
plt.imshow(panoImage)
plt.show()

#bestH2to1=np.matrix([[ 9.71944963e-01,-8.92392029e-03,1.02966138e+02],[-1.63641110e-02,9.70907650e-01,-5.11601289e+00],[-7.22797667e-05,-5.55948484e-05,1.00000000e+00]])
# Q 4.2
panoImage2= imageStitching_noClip(img1,img2,bestH2to1)
plt.subplot(2,1,1)
plt.imshow(panoImage)
plt.subplot(2,1,2)
plt.imshow(panoImage2)
plt.show()

# Q 4.3
img1=skimage.io.imread('../data/incline_L.png')
img2 = skimage.io.imread('../data/incline_R.png')
panoImage3 = generatePanorama(img1, img2)
plt.imshow(panoImage3)
plt.show()

# Q 4.4 (EC)
# Stitch your own photos with your code
img1=skimage.io.imread('../data/myimage1.jpg')
img2 = skimage.io.imread('../data/myimage2.jpg')
panoImage3 = generatePanorama(img1, img2)
plt.imshow(panoImage3)
plt.show()
# Q 4.5 (EC)
# Write code to stitch multiple photos
# see http://www.cs.jhu.edu/~misha/Code/DMG/PNC3/PNC3.zip
# for the full PNC dataset if you want to use that
if False:
    imgs = [skimage.io.imread('../PNC3/src_000{}.png'.format(i)) for i in range(7)]
    panoImage4 = generateMultiPanorama(imgs)
    plt.imshow(panoImage4)
    plt.show()