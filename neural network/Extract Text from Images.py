import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    bboxes = np.asarray(bboxes)
    plt.imshow(bw,cmap = plt.cm.gray)
    for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    plt.tight_layout()
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    row = []
    for i in range(len(bboxes)):
        minr, minc, maxr, maxc = bboxes[i]
        row.append(minr)
    num_row = 0
    iter = row[0]
    lines = []
    m = 0
    for j in range(len(row)):
        if row[j] - iter > 100:# test difference between two value
            each_line = bboxes[m:j]
            num_row += 1
            lines.append(each_line)
            m = j
        if j == len(row)-1:# for last line
            lines.append(bboxes[m:j])
        iter = row[j]
    num_row = num_row+1#for the last row
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])#label
    params = pickle.load(open('q3_weights.pickle','rb'))
    #total_text = []
    #bw = bw[:, ::-1]
    text=''
    for i in range(num_row):
        #total_text = []
        one_line = lines[i] #this line
        one_line = one_line[one_line[:,1].argsort()]
        num_element = len(one_line)# how many element in each row
        diss = np.diff(one_line[:,1])
        threshold = np.mean(diss)
        for j in range(num_element):
            pos = one_line[j,:]
            data = bw[int(pos[0]):int(pos[2]),int(pos[1]):int(pos[3])]
            size_c = pos[3] - pos[1]
            size_r = pos[2] - pos[0]#get size of row and column
            #padding
            if size_c > size_r:
                    pad_size = int(size_c/5)
                    patch_row_1 = (size_c + 2*pad_size - size_r) // 2
                    patch_row_2 = size_c + 2*pad_size - patch_row_1 - size_r
                    data = np.pad(data,[(patch_row_1,patch_row_2),(pad_size,pad_size)],mode = 'constant',constant_values=1)
            else:
                    pad_size = int(size_r/5)
                    patch_column_1 = (size_r + 2*pad_size - size_c) // 2
                    patch_column_2 = size_r + 2*pad_size - patch_column_1 - size_c
                    data = np.pad(data,[(pad_size,pad_size),(patch_column_1,patch_column_2)],mode = 'constant',constant_values=1)
            data = skimage.transform.resize(data,(32,32))
            data = data < data.max()
            data = data==0
            data = np.transpose(data)
            data = data.reshape(1,1024)
            #data = data.flatten()
            h1 = forward(data,params,'layer1')
            probs = forward(h1,params,'output',softmax)
            predict_label = np.argmax(probs,axis = 1)
            predict_label = np.int(predict_label)
            if j != (num_element-1):
                pos_next = one_line[j+1,:]
                if np.abs(pos[1] - pos_next[1]) < 1.5*threshold:#if no need to add space
                    text = text + letters[predict_label]
                else:
                    text = text + letters[predict_label]
                    text = text + ' '#for next
            if j == (num_element-1):
                    text = text + letters[predict_label]
        text = text + '\n'
    print('image',img)
    print(text)