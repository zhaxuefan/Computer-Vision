import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
'''CarSquence=np.load('../data/carseq.npy')
x1=59
x2=145
y1=116
y2=151
rect1=np.load('../data/carseqrects.npy')
number_image = CarSquence.shape[2]
rect = np.asarray([x1,y1,x2,y2])
rects = np.zeros([number_image,4])
rects[0] = rect
It_orin = CarSquence[:,:,0]

for i in range(number_image-1):
        It = CarSquence[:,:,i]
        It1 = CarSquence[:,:,i+1]
        p = LucasKanade.LucasKanade(It, It1, rect)
        rect[0] = rect[0] + p[0]
        rect[1] = rect[1] + p[1]
        rect[2] = rect[2] + p[0]
        rect[3] = rect[3] + p[1]
        print('image number',i)
        rects[i+1,:] = rect
        pd = rects[i+1,:] - rects[0,:]
        pd = np.array([pd[0],pd[1]])
        diff = LucasKanade.LucasKanade(It_orin, It1, rects[0,:],pd)
        diff_p = diff - pd
        #print(diff_p)
        if np.linalg.norm(p - diff_p) < 10:
            p = diff_p
            rect[0] = rect[0] + p[0]
            rect[1] = rect[1] + p[1]
            rect[2] = rect[2] + p[0]
            rect[3] = rect[3] + p[1]
            rects[i+1,:] = rect
        else:
            p = p


        if i % 100==0:
            fig,ax = plt.subplots(1)
            ax1 = ax
            ax.imshow(CarSquence[:,:,i],cmap=plt.get_cmap('gray'))
            mark = patches.Rectangle([rect[0],rect[1]],rect[2]-rect[0],rect[3]-rect[1],linewidth=3,edgecolor='y',facecolor='none')
            mark2=patches.Rectangle([rect1[i,0],rect1[i,1]],rect1[i,2]-rect1[i,0],rect1[i,3]-rect1[i,1],linewidth=3,edgecolor='g',facecolor='none')
            ax.add_patch(mark)
            ax.add_patch(mark2)
            plt.title('frame: %i'%i)
            plt.show()
            print(rect)
#print(rects.shape)
np.save('../data/carseqrects_wcrt.npy', rects)'''

# this is the animation part, please close window before
sylv=np.load('../data/carseq.npy')
rect1=np.load('../data/carseqrects.npy')       # rect Bases rectangle
rect2=np.load('../data/carseqrects_wcrt.npy')    # rect Lucas rectangle
    # rect3=np.load('../data/sylvrect_tc.npy')        # rect with template correction
number_image = sylv.shape[2]
fig,ax = plt.subplots(1)
for i in range(number_image-1):
# rect=np.asarray([x1,y1,x2,y2])
        #if i==1 or i==200 or i==300 or i==350 or i==400:
            #fig,ax = plt.subplots(1)
            ax.imshow(sylv[:,:,i],cmap=plt.get_cmap('gray'))
            mark1=patches.Rectangle([rect1[i,0],rect1[i,1]],rect1[i,2]-rect1[i,0],rect1[i,3]-rect1[i,1],linewidth=3,edgecolor='y',facecolor='none')
            mark2=patches.Rectangle([rect2[i,0],rect2[i,1]],rect2[i,2]-rect2[i,0],rect2[i,3]-rect2[i,1],linewidth=3,edgecolor='g',facecolor='none')
            ax.add_patch(mark1)
            ax.add_patch(mark2)
            plt.title('frame: %i'%i)
            plt.pause(0.01)
            plt.show()
            plt.cla()