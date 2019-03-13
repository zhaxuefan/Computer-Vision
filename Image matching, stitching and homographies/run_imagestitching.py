import skimage.color
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import math
from ec import *
import cv2
#foreground
fg = skimage.io.imread('../data/fg2.png')[:,:,:3]
#fg=skimage.color.rgb2gray(fg)
# background
bg = skimage.io.imread('../data/bg2.png')[:,:,:3]
#bg=skimage.color.rgb2gray(bg)
# binary mask should be grey
mask = skimage.io.imread('../data/mask2.png')[:,:,:3]
mask[:,:,1] = mask[:,:,0]
mask[:,:,2] = mask[:,:,0]

# bad clone
cloned = np.copy(bg)
cloned[np.where(mask >0)] = fg[np.where(mask>0)]

plt.subplot(1,2,1)
plt.imshow(cloned)
plt.title('before')
mask = np.atleast_3d(mask).astype(np.float) / 255.
# Make mask binary
mask[mask != 1] = 0
# Trim to one channel
mask = mask[:,:,0]
channels = fg.shape[-1]
# Call the poisson method on each individual channel
seamless= [poissonStitch(fg[:,:,i], bg[:,:,i], mask) for i in range(channels)]
# Merge the channels back into one image
result = cv2.merge(seamless)
plt.subplot(1,2,2)
plt.imshow(result)
plt.title('after')
plt.show()