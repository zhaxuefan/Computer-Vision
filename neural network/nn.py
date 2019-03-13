import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
	
	W, b = None, None

	W = np.random.uniform(-(np.sqrt(6))/(np.sqrt(in_size+out_size)), \
					   (np.sqrt(6))/(np.sqrt(in_size+out_size)), size=(in_size,out_size))
	b = np.zeros(out_size)
	params['W' + name] = W
	params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
	# y = σ(XW + b)
	#test = sigmoid(np.array([-1000,1000]))
	res = None
	res = np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))
	return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
	"""
	Do a forward pass
	
	Keyword arguments:
		X -- input vector [Examples x D]
		params -- a dictionary containing parameters
		name -- name of the layer
		activation -- the activation function (default is sigmoid)
	"""	
	pre_act, post_act = None, None
	# get the layer parameters
	W = params['W' + name]
	b = params['b' + name]
	
	# your code here
	# ﬁrst hidden layer pre-activation a(1)(x) 
	pre_act = (X@W) + b
	# post-activation values of the ﬁrst hidden layer h(1)(x) 
	post_act = activation(pre_act)
	
	# store the pre-activation and post-activation values
	# these will be important in backprop
	params['cache_' + name] = (X, pre_act, post_act)
	
	return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
	res = None
	e_x = np.exp(x.T - np.max(x, axis = -1))
	res = (e_x / e_x.sum(axis=0)).T
	return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
	loss, acc = None, None
	
	# Cross entropy: Input (N, k) ndarray for both and return scalar (cross entropy)
	# probs: prediction, y: targets
	eps = 1e-12
	probs = np.clip(probs, eps, 1. - eps)
	N = probs.shape[0]
	loss = -np.sum(y*np.log(probs+1e-9)) /N
	
	diff = np.argmax(probs, axis=1) - np.argmax(y, axis=1) # zero is correct
	acc = np.count_nonzero(diff==0) / len(diff)
	
	
	return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
	res = post_act*(1.0-post_act)
	return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
	"""
	Do a backwards pass
	
	Keyword arguments:
		delta -- errors to backprop
		params -- a dictionary containing parameters
		name -- name of the layer
		activation_deriv -- the derivative of the activation_func
	"""
	
	grad_X, grad_W, grad_b = None, None, None
	# everything you may need for this layer
	W = params['W' + name]
	b = params['b' + name]
	X, pre_act, post_act = params['cache_' + name]
	# your code here
	# do the derivative through activation first
	# then compute the derivative W,b, and X
	
	if name == 'output':
		grad_W = X.T @ delta
		grad_b = np.mean(delta, axis=0)
		grad_X = delta @ W.T
	else:
		delta_h = activation_deriv(post_act)*delta # hidden error
		grad_W = X.T @ delta_h
		grad_b = np.mean(delta_h, axis=0)
		grad_X = delta_h @ W.T
	
	# store the gradients
	params['grad_W' + name] = grad_W
	params['grad_b' + name] = grad_b
	return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
	bnum = int(len(x) / batch_size) # number of batches (length/batch_size)
	batches = [0]*bnum
	pos = np.arange(x.shape[0]).reshape((x.shape[0],1))
	stack = np.hstack((x,y,pos))
	sf = stack # shuffle
	np.random.shuffle(sf)
	x_sf = sf[:,:x.shape[1]]
	y_sf = sf[:,x.shape[1]:-1]
	b_pos = sf[:,-1:]
	b_pos = b_pos[:,0]
	for i in range(bnum):
		batches[i] = (x_sf[i*batch_size:(i+1)*batch_size,:],y_sf[i*batch_size:(i+1)*batch_size,:])
		
	return batches,b_pos
