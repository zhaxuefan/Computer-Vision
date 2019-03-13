import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 200
# pick a batch size, learning rate
batch_size = 30
learning_rate = 1e-2
hidden_size = 64

batches,bpos = get_random_batches(train_x,train_y,batch_size)
#batches = get_random_batches(valid_x, valid_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,36,params,'output')
accuracy = []
loss = []
valid_accuracy = []
valid_loss = []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    acc = []
    predict = np.zeros([train_y.shape[0],train_y.shape[1]])
    bat = 0
    #probs = []
    for xb,yb in batches:
        pass
        # training loop can be exactly the same as q2!
        batch_h1 = forward(xb,params,'layer1')
        batch_probs = forward(batch_h1,params,'output',softmax)
        # loss
        #probs.append(batch_probs)
        # be sure to add loss and accuracy to epoch totals
        batch_loss, batch_acc = compute_loss_and_acc(yb, batch_probs)
        total_loss = total_loss + batch_loss
        acc.append(batch_acc)
        batch_delta1 = batch_probs
        #confusion matrix
        predict[bpos[bat*batch_size:(bat+1)*batch_size].astype(int)] = batch_probs 
        bat = bat+1
        yb_idx=np.where(yb == 1)[1]
        batch_delta1[np.arange(batch_probs.shape[0]),yb_idx] -= 1
        # backward
        batch_delta2 = backwards(batch_delta1,params,'output',linear_deriv)
        backwards(batch_delta2,params,'layer1',sigmoid_deriv)
        # apply gradient
        params['Wlayer1'] = params['Wlayer1'] - learning_rate*params['grad_Wlayer1']
        params['Woutput'] = params['Woutput'] - learning_rate*params['grad_Woutput']
        params['blayer1'] = params['blayer1'] - learning_rate*params['grad_blayer1']
        params['boutput'] = params['boutput'] - learning_rate*params['grad_boutput']
    total_acc = np.mean(acc)
    accuracy.append(total_acc)
    loss.append(total_loss)
    #calculate validation
    val_h1 = forward(valid_x,params,'layer1')
    val_probs = forward(val_h1,params,'output',softmax)
    # be sure to add loss and accuracy to epoch totals
    val_loss, val_acc = compute_loss_and_acc(valid_y,val_probs)
    valid_loss.append(val_loss)
    valid_accuracy.append(val_acc)
    print('Validation accuracy: ',val_acc)#,'Validation loss:', val_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
# run on validation set and report accuracy! should be above 75%
#valid_acc = None
epoch = np.linspace(1.0,200.0,num = 200)
plt.figure(1)
plt.plot(epoch,accuracy,label = 'train accuracy')
plt.plot(epoch,valid_accuracy,label = 'valid accuracy')
plt.title('accuracy')
plt.legend()

plt.figure(2)
plt.plot(epoch,loss,label = 'train loss')
plt.plot(epoch,valid_loss,label = 'valid loss')
plt.title('loss')
plt.legend()
plt.show()
valid_acc = valid_accuracy[-1]
print('Validation accuracy: ',valid_acc)
#test
test_h1 = forward(test_x,params,'layer1')
test_probs = forward(test_h1,params,'output',softmax)
test_loss, test_acc = compute_loss_and_acc(test_y,test_probs)
print(test_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
first_weight = params['W' + 'layer1']
[size_image,size_hidden] = first_weight.shape
plt.figure(3)
fig = plt.figure(3, (6., 6.))
grid = ImageGrid(fig,111,nrows_ncols=(8,8),axes_pad=0.01)
for i in range(size_hidden):
    im = first_weight[:,i].reshape(32,32)
    grid[i].imshow(im)  # The AxesGrid object work as a list of axes.
plt.show()
plt.figure(4)
fig1 = plt.figure(4, (6., 6.))
grid1 = ImageGrid(fig1,111,nrows_ncols=(8,8),axes_pad=0.01)
for i in range(size_hidden):
    limit = np.sqrt(6.0 / (1024 + hidden_size))
    initial_weight = np.random.uniform(-limit,limit,[1024,hidden_size])
    im1 = initial_weight[:,i].reshape(32,32)
    grid1[i].imshow(im1)  # The AxesGrid object work as a list of axes.
plt.show()


# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
'''predict = np.zeros([train_y.shape[0],train_y.shape[1]])
for i in range(len(probs)):
    predict[i*batch_size:i*batch_size+batch_size,:] = probs[i]
for j in range(predict.shape[0]):
    max = predict[j,:].max()
    for i1 in range(predict.shape[1]):
        if predict[j,i1] == max:
            predict[j,i1] = 1
        else:
            predict[j,i1] = 0
for ii in range(total_yb.shape[1]):
    for ij in range(total_yb.shape[1]):
        if predict[ii,ij] == total_yb[ii,ij]:
            confusion_matrix[ii,ij] += 1'''
for i in range(train_y.shape[0]):
    loc = np.int(np.where(train_y[i,:]==1)[0])
    loc_predict = np.int(np.where(predict[i,:] == max(predict[i,:]))[0])
    confusion_matrix[loc,loc_predict] += 1
        
import string
plt.figure(5)
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()