import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import skimage.transform
import scipy.ndimage
#import cv2
import LucasKanadeAffine

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    Aerial=np.load('../data/aerialseq.npy')
    number_image = Aerial.shape[2]
    rects=np.zeros(Aerial.shape)
    for i in range(number_image-1):
        It = Aerial[:,:,i]
        It1 = Aerial[:,:,i+1]
        mask=SubtractDominantMotion(It,It1)
        print('frame:',i)
        I_tack_P=It1.copy()
        I_tack_P[mask!=0]=1
        rects[:,:,i] = I_tack_P
        I_tack=np.dstack((It1,It,I_tack_P))
        if i==30 or i==60 or i==90 or i==120:
        # if i%5==0:
            fig,ax = plt.subplots(1)
            ax.imshow(I_tack),plt.title('It')
            plt.title('frame: %i'%i)
            plt.show()
np.save('../data/aerialseqrects.npy', rects)

#this is for animation
if __name__ == '__main__':
    Aerial=np.load('../data/aerialseq.npy')
    I_tack_P = np.load('../data/aerialseqrects.npy')
    number_image = Aerial.shape[2]
    #rects = np.zeros(Aerial.shape)
    fig,ax = plt.subplots(1)
    for i in range(number_image-1):
        It = Aerial[:,:,i]
        It1 = Aerial[:,:,i+1]
        #mask = SubtractDominantMotion(It,It1)
        print('frame:',i)
        #I_tack_P=It1.copy()
        #I_tack_P[mask!=0]=1
        I_tack=np.dstack((It1,It,I_tack_P[:,:,i]))
        
        ax.imshow(I_tack),plt.title('It')
        #plt.imshow(I_tack),plt.title('It')
        plt.title('frame: %i'%i)
        #plt.show()
        plt.pause(0.01)
        plt.show()
        plt.cla()