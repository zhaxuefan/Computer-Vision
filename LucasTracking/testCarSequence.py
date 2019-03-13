import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
CarSquence=np.load('../data/carseq.npy')
x1=59
x2=145
y1=116
y2=151

number_image = CarSquence.shape[2]
rect = np.asarray([x1,y1,x2,y2])
rects = np.zeros([number_image,4])
rects[0] = rect

for i in range(number_image-1):
        It = CarSquence[:,:,i]
        It1 = CarSquence[:,:,i+1]

        p = LucasKanade.LucasKanade(It, It1, rect)
        x1 = x1+p[0]
        x2 = x2+p[0]
        y1 = y1+p[1]
        y2 = y2+p[1]
        print('image number',i)
        rect = np.array([x1,y1,x2,y2])
        rects[i+1,:] = rect


        if i%100==0:
            fig,ax = plt.subplots(1)
            ax.imshow(CarSquence[:,:,i],cmap=plt.get_cmap('gray'))
            mark=patches.Rectangle([rect[0],rect[1]],rect[2]-rect[0],rect[3]-rect[1],linewidth=3,edgecolor='g',facecolor='none')
            ax.add_patch(mark)
            plt.title('frame: %i'%i)
            plt.show()
            print(rect)
#print(rects.shape)
np.save('../data/carseqrects.npy', rects)