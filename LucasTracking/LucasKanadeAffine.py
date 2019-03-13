import numpy as np
from scipy.interpolate import RectBivariateSpline
#import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
#-----------------To do------------------
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0,0,1]])

# initialize parameters
    threshold=10e-2
    tre_p=10
    p0=np.array([0,0,0,0,0,0])
    # gradient= np.asarray(np.gradient(It1))
    # X_gradient=gradient[1]
    # Y_gradient=gradient[0]
    H,W=It.shape
    X_range=np.arange(W)
    Y_range=np.arange(H)
    Spline=RectBivariateSpline(Y_range,X_range,It)
    Spline1=RectBivariateSpline(Y_range,X_range,It1)
    X,Y=np.meshgrid(X_range,Y_range)
    X=X.flatten()
    Y=Y.flatten()

    Interation=0
    while tre_p>threshold:
        Interation=Interation+1
        # print(Interation)
        M=np.array([[1+p0[0], p0[1],    p0[2] ],
                   [p0[3],    1+p0[4],  p0[5] ],
                   [0,        0,        1.    ]])

        X_w=X*M[0,0]+Y*M[0,1]+M[0,2]
        Y_w=X*M[1,0]+Y*M[1,1]+M[1,2]

        mask_valid=np.logical_and(np.logical_and(X_w>=0,X_w<It.shape[1]),np.logical_and(Y_w>=0,Y_w<It.shape[0]))

        X_valid=X[mask_valid]
        Y_valid=Y[mask_valid]
        X1_valid=X_w[mask_valid]
        Y1_valid=Y_w[mask_valid]
        D_x=Spline1(Y1_valid,X1_valid,0,1,grid=False).flatten()
        D_y=Spline1(Y1_valid,X1_valid,1,0,grid=False).flatten()

        SDQ=np.stack([D_x*X1_valid,
                      D_x*Y1_valid,
                      D_x,
                      D_y*X1_valid,
                      D_y*Y1_valid,
                      D_y],axis=1)
        H = np.dot(np.transpose(SDQ),SDQ)   # Hessian matrix: 2*2

        template=Spline(Y_valid,X_valid,grid=False)
        It1_w=Spline1(Y1_valid,X1_valid,grid=False)

        Error=template-It1_w   #N*1
        B= SDQ.T.dot(Error.flatten())    # 6*1 = 6*N * N*1
        delta_p =np.dot(np.linalg.inv(H),B)  # delta p: 6*1[ px ; py]
        p0 = p0+delta_p
        tre_p = np.linalg.norm(delta_p)
        # print('normal(delta_p)',tre_p)
    # print('affine finished')
    return M

'''if __name__ == '__main__':
    Aerial=np.load('../data/aerialseq.npy')
    number_image = Aerial.shape[2]
    rects=np.zeros([number_image,4])
    for i in range(number_image-1):
        It = Aerial[:,:,i]
        It1 = Aerial[:,:,i+3]
#        m=SubtractDominantMotion(It,It1)
        print(i)
        W=LucasKanadeAffine(It,It1)
        print(W)
        It_w= cv2.warpPerspective(It1,W,(It1.shape[1],It1.shape[0]))
        plt.subplot(1,3,1),plt.imshow(It),plt.title('It')
        plt.subplot(1,3,2),plt.imshow(It_w),plt.title('It_w')
        plt.subplot(1,3,3),plt.imshow(It1),plt.title('It1')
        plt.show()'''