import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import skimage.io
import scipy
import scipy.signal
import skimage.color
import math

from q2 import eightpoint
from q3 import essentialMatrix, triangulate
from util import camera2
# Q 4.1
# np.pad may be helpful
def epipolarCorrespondence(im1, im2, F, x1, y1):
    def gaussian_2d_kernel(nn, sigma=5):
        a = np.asarray([[x**2 + y**2 for x in range(-nn,nn+1)] for y in range(-nn,nn+1)])
        kernel=np.exp(-a/(2*sigma**2))
        return kernel
    m,n = im2.shape[:2]
    patch = 5 # patch size
    img1,img2 = skimage.color.rgb2gray(im1),skimage.color.rgb2gray(im2)
    weighting = gaussian_2d_kernel(patch,5)
    pt = np.array([x1,y1,1])
    epline = np.dot(F,pt) # epipolar line on im2
    t = np.linspace(1,m,m)
    lt = np.array([(epline[2]+epline[1]*tt)/(-epline[0]) for tt in t])
    lt=lt.flatten()
    ndx = (lt>=0) & (lt<n)
    ndx=np.sqrt((lt-x1)**2+(t-y1)**2)<30
    ndx=ndx.flatten()
    t=t[ndx]
    lt=lt[ndx]
    padimg1=np.pad(img1,patch,'constant')
    padx1=int(x1+patch)
    pady1=int(y1+patch)
    Patch_im1=padimg1[pady1-patch:pady1+patch+1,padx1-patch:padx1+patch+1]
    padimg2=np.pad(img2,patch,'constant')
    padt=t+patch
    padlt=lt+patch
    num=t.shape[0]
    respone=np.ones(num)*np.inf
    for k in range(num): #对于每一个候选点
        yy=int(padt[k])
        xx=int(np.round(padlt[k]))
        Patch_im2=padimg2[yy-patch:yy+patch+1,xx-patch:xx+patch+1]

        respone[k]=np.sum(abs(Patch_im2-Patch_im1)*weighting)
    idx=np.argmin(respone)
    y2=t[idx]
    x2=int(np.round(lt[idx]))
    return x2, y2

# Q 4.2
# this is the "all in one" function that combines everything
def visualize(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):
    # you'll want a roughly cubic meter
    # around the center of mass
    im1 = skimage.io.imread(IM1_PATH)
    im2 = skimage.io.imread(IM2_PATH)
    corr = scipy.io.loadmat(TEMPLE_CORRS)
    x1= corr['x1']
    y1= corr['y1']
    num=x1.shape[0]
    x2=np.zeros(num)
    y2=np.zeros(num)
    for idx in range(num):
        x,y = epipolarCorrespondence(im1,im2,F,x1[idx],y1[idx])
        x2[idx]=x
        y2[idx]=y
    pts1= np.hstack((x1,y1))
    pts2= np.vstack((x2,y2)).transpose()
    E = essentialMatrix(F,K1,K2)
    E = E/E[2,2]
    M2s = camera2(E)
    C1 = np.hstack([np.eye(3),np.zeros((3,1))])
    for C2 in M2s:
        P, err = triangulate(K1.dot(C1),pts1,K2.dot(C2),pts2)
        if(P.min(0)[2] > 0):
            # we're the right one!
            break
    P, err = triangulate(K1.dot(C1),pts1,K2.dot(C2),pts2)
    scipy.io.savemat('q4_2.mat', {'P':P,'M1':C1,'M2':C2,'K1':K1,'K2':K2})
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim
    ax.set_ylim
    ax.set_zlim
    ax.set_aspect('equal')
    ax.scatter(P[:,0],P[:,1],P[:,2])
    plt.show()



# Extra credit
def visualizeDense(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):
    return