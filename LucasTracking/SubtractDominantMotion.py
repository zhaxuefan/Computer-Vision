import numpy as np
import LucasKanadeAffine
import scipy.ndimage
#import cv2
import sklearn.preprocessing
import matplotlib.pyplot as plt
import skimage.transform
import skimage.morphology
from scipy.interpolate import RectBivariateSpline
from InverseCompositionAffine import InverseCompositionAffine
def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    # mask = np.ones(image1.shape, dtype=bool)
    # M=LucasKanadeAffine.LucasKanadeAffine(image1, image2)
    threshold = 0.2
    M=LucasKanadeAffine.LucasKanadeAffine(image1,image2)

    # test Inverse Composition Affine
    #M=InverseCompositionAffine(image1,image2)
    H,W=image1.shape
    X_range=np.arange(W)
    Y_range=np.arange(H)
    Spline=RectBivariateSpline(Y_range,X_range,image1)
    Spline1=RectBivariateSpline(Y_range,X_range,image2)
    X,Y=np.meshgrid(X_range,Y_range)
    X=X.flatten()
    Y=Y.flatten()

    X_w=(X*M[0,0]+Y*M[0,1]+M[0,2]).reshape(H,W)
    Y_w=(X*M[1,0]+Y*M[1,1]+M[1,2]).reshape(H,W)
    template=Spline.ev(Y.reshape(H,W),X.reshape(H,W))
    image2_w=Spline1.ev(Y_w,X_w)
    diff = abs(template - image2_w)
    mask = sklearn.preprocessing.binarize(diff,threshold)
    selem=skimage.morphology.disk(3)
    mask1 = skimage.morphology.binary_dilation(mask,selem)
    # plt.subplot(2,1,1),plt.imshow(mask)
    # plt.subplot(2,1,2),plt.imshow(mask1)
    # plt.show()
    mask = mask1
    return mask
