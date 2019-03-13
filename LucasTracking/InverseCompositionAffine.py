import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# initialize paramzeters
    threshold=0.01
    tre_p=10
    H,W=It.shape
    X_range=np.arange(W)
    Y_range=np.arange(H)
    Spline=RectBivariateSpline(Y_range,X_range,It)
    Spline1=RectBivariateSpline(Y_range,X_range,It1)
    X,Y=np.meshgrid(X_range,Y_range)
    X=X.flatten()
    Y=Y.flatten()

    gradient= np.asarray(np.gradient(It))
    D_x=gradient[1].flatten()
    D_y=gradient[0].flatten()
    SDQ=np.stack([D_x*X,
                D_x*Y,
                D_x,
                D_y*X,
                D_y*Y,
                D_y],axis=1)
    H = np.dot(np.transpose(SDQ),SDQ)

    Interation=0
    while tre_p>threshold:

        X_w=X*M[0,0]+Y*M[0,1]+M[0,2]
        Y_w=X*M[1,0]+Y*M[1,1]+M[1,2]

        mask_valid=np.logical_and(np.logical_and(X_w>=0,X_w<It.shape[1]),np.logical_and(Y_w>=0,Y_w<It.shape[0]))

        X_valid=X[mask_valid]
        Y_valid=Y[mask_valid]
        X1_valid=X_w[mask_valid]
        Y1_valid=Y_w[mask_valid]
        SDQ_m=SDQ[mask_valid,:]

        It1_w=Spline1(Y1_valid,X1_valid,grid=False)
        Template=Spline(Y_valid,X_valid,grid=False)

        Error=It1_w-Template   #N*1

        B= SDQ_m.T.dot(Error.flatten())    # 6*1 = 6*N * N*1
        delta_p =np.dot(np.linalg.inv(H),B)  # delta p: 6*1[ px ; py]
        tre_p = np.linalg.norm(delta_p)
        Interation=Interation+1
        # print('print interation times:',Interation,'delta_p',tre_p)
        P=np.array([[1+delta_p[0], delta_p[1],    delta_p[2] ],
                   [delta_p[3],    1+delta_p[4],  delta_p[5] ],
                   [0,        0,        1.    ]])
        M=M[0:2,:].dot(np.linalg.inv(P))
        M=np.append(M,[[0,0,1]],axis=0)
    # print('affine finished')
    return M
