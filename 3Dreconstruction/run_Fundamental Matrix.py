from q2 import eightpoint, sevenpoint
from util import plot_epipolar_lines

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
IM1_PATH = os.path.join(HOMEWORK_DIR,'data','im1.png')
IM2_PATH = os.path.join(HOMEWORK_DIR,'data','im2.png')

im1 = skimage.io.imread(IM1_PATH)
im2 = skimage.io.imread(IM2_PATH)
np.set_printoptions(suppress=True)

# Q2.1
corr = scipy.io.loadmat(SOME_CORRS)
pts1 = corr['pts1']
pts2 = corr['pts2']
idxs = np.array([82,19,56,84,54,24,18])

if PARTS_RUN&1 > 0:
    F = eightpoint(pts1,pts2,max(im1.shape))
    F = F/F[2,2]
    # fundamental matrix must have rank 2!
    assert(np.linalg.matrix_rank(F) == 2)
    print(F)
    plot_epipolar_lines(im1,im2,F,pts1,idxs)

# Q2.2
if PARTS_RUN&2 > 0:
    # feel free to change, seven points
    idxs = np.arange(pts1.shape[0])
    idxs = np.random.choice(idxs,7,False)
    rank=3
    while rank!=2:
        F = sevenpoint(pts1[idxs,:],pts2[idxs,:],max(im1.shape))
        rank=np.linalg.matrix_rank(F)
    F = F/F[2,2]
    print(F)
    plot_epipolar_lines(im1,im2,F,pts1,idxs)