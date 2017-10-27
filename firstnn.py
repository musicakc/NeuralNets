'''
Neural Networks Tutorial 1:
https://iamtrask.github.io/2015/07/12/basic-python-network/
'''

import numpy as np

#sigmoid function for calculating hidden layers
def nonlin(x,deriv=False):
	if(deriv == True):
		return x*(1-x)
	return 1/(1 + np.exp(-x)) #formula of sigmoid fuction

#input
x = np.array([[0,0,1], [0,1,1],[1,0,1],[1,1,1]])

#output
y = np.array([[0,0,1,1]]).T #transpose the output array

#seed random numbers
np.random.seed(1)

#initialise weights randomly
syn0 = 2 * np.random.random((3,1)) - 1

for i in range(10000):
	#forward propogation
	l0 = x
	#hidden layer l1
	l1 = nonlin(np.dot(l0,syn0))

	#calculate error between output and l1
	l1_error = y - l1

	l1_delta = l1_error * nonlin(l1,True)

print (l1)
