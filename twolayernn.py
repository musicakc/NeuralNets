'''
Neural Networks Tutorial 1:
https://iamtrask.github.io/2015/07/12/basic-python-network/

3 input nodes, 4 training examples
'''

import numpy as np

'''
Nonlinearity or sigmoid function used to map a value to a 
value between 0 and 1
'''
def nonlin(x,deriv=False):
	#ouput can be used to create derivative
	if(deriv == True):
		return x*(1-x) #slope or derivative
	return 1/(1 + np.exp(-x)) #formula of sigmoid fuction

#input x_(4*3)
x = np.array([[0,0,1], [0,1,1],[1,0,1],[1,1,1]])

#output y_(4*1)
y = np.array([[0,0,1,1]]).T #transpose the output array

#seed random numbers
np.random.seed(1)

'''
Initialise weights randomly with mean 0, 
of dimension [3*1] for 3 inputs and 1 ouput
'''
syn0 = 2 * np.random.random((3,1)) - 1

#iterate multiple times to optimise our neural network
for i in range(10000):
	#forward propogation
	l0 = x
	#hidden layer l1_(4*1) = [l0_(4*3) dot syn0_(3*1)]
	l1 = nonlin(np.dot(l0,syn0))

	#calculate error between output and l1
	l1_error = y - l1

	#slope of l1 * error to reduce error of high confidence predictions
	l1_delta = l1_error * nonlin(l1,True)

	#update weights using l1_delta
	syn0 += np.dot(l0.T, l1_delta)

print (l1)
