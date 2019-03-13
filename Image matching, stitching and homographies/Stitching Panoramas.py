import numpy as np
from q2 import *
from q3 import *
import skimage.color
import numpy as np

# you may find this useful in removing borders
# from pnc series images (which are RGBA)
# and have boundary regions
def clip_alpha(img):
    img[:,:,3][np.where(img[:,:,3] < 1.0)] = 0.0
    return img 

# Q 4.1
# this should be a less hacky version of
# composite image from Q3
# img1 and img2 are RGB-A
# warp order 0 might help
# try warping both images then combining
# feel free to hardcode output size
def imageStitching(img1, img2, H2to1):
    panoImg = None
    # YOUR CODE HERE
    panoImg=np.zeros((img2.shape[0],img2.shape[1]+img1.shape[1]))
    xh = np.linalg.inv(H2to1)
    #xh=xh/xh[2,2]
    dsize=panoImg.shape
    #print(np.asarray([xshape,yshape,zshape]))
    #dsize=(img2.shape[0],img2.shape[1],3)
    wrapImg=skimage.transform.warp(img2,xh,output_shape=dsize)
    panoImg=wrapImg
    panoImg[0:img1.shape[0],0:img1.shape[1]]=img1*255
    return panoImg


# Q 4.2
# you should make the whole image fit in that width
# python may be inv(T) compared to MATLAB
def imageStitching_noClip(img1, img2, H2to1, panoWidth=1280):
    panoImg = None
    # YOUR CODE HERE
    height=panoWidth*np.max([img1.shape[0],img2.shape[0]])/(img1.shape[0]+img2.shape[0])
    panoImg=np.zeros((np.int(height),panoWidth))
    n=panoWidth/height
    print(n)
    dsize=panoImg.shape
    print(type(dsize))
    xh = np.linalg.inv(H2to1)
    #t=np.min(img1.shape[0],img2.shape[0])
    #t2=np.min(img1.shape[1],img2.shape[1])
    M=[[n,0,xh[0,2]/xh[2,2]/n],[0,n,xh[1,2]/xh[2,2]/n],[0,0,1]]
    #M=[[n,0,],[0,n,t],[0,0,1]]
    Mh=np.linalg.inv(M)
    warp_im1=skimage.transform.warp(img1,Mh,output_shape=dsize)
    warp_im2 = skimage.transform.warp(img2,np.dot(xh,Mh),output_shape=dsize);
    panoImg=warp_im2
    print(panoImg.shape)
    panoImg[:,0:img1.shape[1]]=warp_im1[:,0:img1.shape[1]]
    return panoImg

# Q 4.3
# should return a stitched image
# if x & y get flipped, np.flip(_,1) can help
def generatePanorama(img1, img2):
    panoImage = None
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
    #bestH2to1=np.matrix([[ 6.54942195e-01,-4.61108416e-02,3.66373752e+02],[-7.65620558e-02,8.70338266e-01,-1.61309760e+01],[-3.50126115e-04,-2.01507101e-05,1.00000000e+00]])
    panoImage = imageStitching(img1,img2,bestH2to1)
    return panoImage

# Q 4.5
# I found it easier to just write a new function
# pairwise stitching from right to left worked for me!
def generateMultiPanorama(imgs):
    panoImage = None
    # YOUR CODE HERE
    return panoImage
