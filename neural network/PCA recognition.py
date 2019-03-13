import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

# we don't need labels now!
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

dim = 32
# do PCA
[U,s,Vh] = np.linalg.svd(train_x)
#principle components is U*S(scores)
V = Vh.T
# rebuild a low-rank version
lrank = None
lrank = dim

#projection_matrix = np.dot(U[:, :lrank], np.dot(S, V[:,:lrank].T))
projection_matrix = V[:,:lrank] @ V[:,:lrank].T
# rebuild it
recon = None
#recon = np.dot(projection_matrix,V_test[:,:lrank].T)
recon = np.dot(test_x, projection_matrix)
idx = [1,20,152,162,310,320,501,520,660,670]
recon = recon[idx]
test_x = test_x[idx]
for i in range(10):
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(test_x[i].reshape(32,32).T)
    #plt.imshow(test_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon[i].reshape(32,32).T)
    plt.show()

# build valid dataset
recon_valid = None
total_psnr = 0
recon_valid = np.dot(valid_x, projection_matrix)
total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
total_psnr = np.array(total).mean()
print(np.array(total).mean())