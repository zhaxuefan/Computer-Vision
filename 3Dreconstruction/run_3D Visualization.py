from q2 import eightpoint
from q4 import epipolarCorrespondence,visualize, visualizeDense
from util import plot_matched_points

import sys
import os
import numpy as np
import scipy.io
import skimage.io
import matplotlib.pyplot as plt

# Setup
# can take homework_dir as first argument
HOMEWORK_DIR = ".." if len(sys.argv) < 2 else sys.argv[1]
PARTS_RUN = 15 if len(sys.argv) < 3 else int(sys.argv[2])
SOME_CORRS = os.path.join(HOMEWORK_DIR,'data','some_corresp.mat')
TEMPLE_CORRS = os.path.join(HOMEWORK_DIR,'data','templeCoords.mat')
INTRINS = os.path.join(HOMEWORK_DIR,'data','intrinsics.mat')

IM1_PATH = os.path.join(HOMEWORK_DIR,'data','im1.png')
IM2_PATH = os.path.join(HOMEWORK_DIR,'data','im2.png')

im1 = skimage.io.imread(IM1_PATH)
im2 = skimage.io.imread(IM2_PATH)
np.set_printoptions(suppress=True)

# Q4.1
corr = scipy.io.loadmat(SOME_CORRS)
pts1 = corr['pts1']
pts2 = corr['pts2']
idxs = np.array([82,19,56,84,54,24,18,104])

if PARTS_RUN&1 > 0:
    F = eightpoint(pts1,pts2,max(im1.shape))
    #F = eightpoint(pts1[idxs,:],pts2[idxs,:],max(im1.shape))
    F = F/F[2,2]
    pts2e = []
    for idx in idxs:
        p2e = epipolarCorrespondence(im1,im2,F,pts1[idx,0],pts1[idx,1])
        pts2e.append(p2e)
    #pts2e = np.array(pts2e)
    print(pts2e-pts2[idxs,:])
    scipy.io.savemat('q2_6',{'p1':pts1[idxs,:],'p2':pts2[idxs,:],'F':F})
    print('Matching error: {:.2f}'.format(np.linalg.norm(pts2e-pts2[idxs,:])))
    plot_matched_points(im1,im2,F,pts1[idxs,:],pts2[idxs,:],pts2e)
    
# Q4.2
if PARTS_RUN&2 > 0:
    import h5py
    with h5py.File(INTRINS, 'r') as f:
        K1 = np.array(f['K1']).T
        K2 = np.array(f['K2']).T
    F = eightpoint(pts1,pts2,max(im1.shape))
    F = F/F[2,2]
    visualize(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2)

# Q4.3 (Extra Credit)
if PARTS_RUN&4 > 0:
    visualizeDense(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2)

