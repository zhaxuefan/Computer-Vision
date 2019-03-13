import sys
import os
import numpy as np
import scipy.io
import skimage.io
import matplotlib.pyplot as plt
from util import camera2
from q2 import eightpoint, sevenpoint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from q3 import essentialMatrix,triangulate
from q5 import ransacF,rodrigues, invRodrigues, rodriguesResidual, bundleAdjustment
from q5 import compact

# Setup
# can take homework_dir as first argument
HOMEWORK_DIR = ".." if len(sys.argv) < 2 else sys.argv[1]
PARTS_RUN = 15 if len(sys.argv) < 3 else int(sys.argv[2])
SOME_CORRS = os.path.join(HOMEWORK_DIR,'data','some_corresp_noisy.mat')
INTRINS = os.path.join(HOMEWORK_DIR,'data','intrinsics.mat')
IM1_PATH = os.path.join(HOMEWORK_DIR,'data','im1.png')
IM2_PATH = os.path.join(HOMEWORK_DIR,'data','im2.png')

im1 = skimage.io.imread(IM1_PATH)
im2 = skimage.io.imread(IM2_PATH)
np.set_printoptions(suppress=True)

import h5py
with h5py.File(INTRINS, 'r') as f:
    K1 = np.array(f['K1']).T
    K2 = np.array(f['K2']).T

# Q5.1
corr = scipy.io.loadmat(SOME_CORRS)
pts1 = corr['pts1']
pts2 = corr['pts2']
F_reference=eightpoint(pts1,pts2,max(im1.shape))

F,inliers = ransacF(pts1, pts2, max(im1.shape))
F = F/F[2,2]
print(F)
print(inliers)
# Q5.2
r = np.array([0,2,0])
R = rodrigues(r)
print(R)
print('both the below should be identity')
print(R.T.dot(R))
print(R.dot(R.T))

# Q5.2
r = invRodrigues(R)
print('should be r')
print(r)

# Q5.3
E = essentialMatrix(F,K1,K2)
E = E/E[2,2]
M2s = camera2(E)

goodP1 = pts1[inliers]
goodP2 = pts2[inliers]

C1 = np.hstack([np.eye(3),np.zeros((3,1))])
for C2 in M2s:
    P, err = triangulate(K1.dot(C1),goodP1,K2.dot(C2),goodP2)
    if(P.min(0)[2] > 0):
        # we're the right one!
        break
print('original error is ',err)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim
ax.set_ylim
ax.set_zlim
ax.set_aspect('equal')
ax.scatter(P[:,0],P[:,1],P[:,2])
plt.show()
# you should create this from the above (using P,C2)
initialx = compact(P,C2[:,0:-1],C2[:,-1])
err = rodriguesResidual(K1,C1,goodP1,K2,goodP2,initialx)
print('initial error is ',err)
C2n,Pn = bundleAdjustment(K1,C1,goodP1,K2,C2,goodP2,P)
# you should create this from the above (using P,C2)
finalx =  compact(Pn,C2n[:,0:-1],C2n[:,-1])
err = rodriguesResidual(K1,C1,goodP1,K2,goodP2,finalx)
print('final error is ',err)
fig1 = plt.figure()
ax = Axes3D(fig1)
ax.set_xlim
ax.set_ylim
ax.set_zlim
ax.set_aspect('equal')
ax.scatter(P[:,0],P[:,1],P[:,2])
plt.show()


