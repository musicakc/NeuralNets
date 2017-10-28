'''
Neural Networks Tutorial 1:
https://iamtrask.github.io/2015/07/12/basic-python-network/

3 input nodes, 4 training examples
'''

import numpy as np

alphas = [0.001,0.01,0.1,1,10,100,1000]
'''
Nonlinearity or sigmoid function used to map a value to a 
value between 0 and 1
'''
def nonlin(x):
	return 1/(1 + np.exp(-x)) #formula of sigmoid fuction

def derivative(op):
	return op * (1-op)

#input x_(4*3)
x = np.array([[0,0,1], [0,1,1],[1,0,1],[1,1,1]])

#output y_(4*1)
y = np.array([[0,1,1,0]]).T #transpose the output array

for alpha in alphas:
	print ("\nTraining Alphas: " + str(alpha))
	np.random.seed(1)


	'''
	Initialise weights randomly with mean 0, 
	of dimension [3*1] for 3 inputs and 1 ouput
	'''
	syn0 = 2 * np.random.random((3, 4)) - 1
	syn1 = 2 * np.random.random((4, 1)) - 1

	prev_syn0_wt = np.zeros_like(syn0)
	prev_syn1_wt = np.zeros_like(syn1)

	syn0_count = np.zeros_like(syn0)
	syn1_count = np.zeros_like(syn1)
	#print(syn0_count)
	#print(syn1_count)

	#iterate multiple times to optimise our neural network
	for i in range(60000):

		#forward propogation
		l0 = x
		#hidden layer l1_(4*4) = [l0_(4*3) dot syn0_(3*4)]
		l1 = nonlin(np.dot(l0,syn0))
		#hidden layer l2_(4*1) = [l1_(4*4) dot syn1_(4*1)]
		l2 = nonlin(np.dot(l1,syn1))

		#calculate error between output and l2
		l2_error = y - l2

		if ((i % 10000) == 0):
			print ("Error after " + str(i) + " iterations: " + str(np.mean(np.abs(l2_error))))

		#slope of l2 * error to reduce error of high confidence predictions
		l2_delta = l2_error * derivative(l2)

		'''
		calculate confidence weighted error by calculating how much
		each node value contributed to the error in l2
		'''
		l1_error = l2_delta.dot(syn1.T)

		#slope of l1 * error to reduce error of high confidence predictions
		l1_delta = l1_error * derivative(l1)

		syn1_wt = l1.T.dot(l2_delta)
		syn0_wt = l0.T.dot(l1_delta)
		#print(syn0_wt)
		#print(syn1_wt)

		if (i > 0):
			syn0_count = np.abs(((syn0_wt) + 0) - ((prev_syn0_wt) + 0))
			syn1_count = np.abs(((syn1_wt) + 0) - ((prev_syn1_wt) + 0))
		#print(syn0_count)
		#print(syn1_count)		

		#update weights using syn1_wt
		syn1 += alpha * syn1_wt
		#update weights using syn0_wt
		syn0 += alpha * syn0_wt

		prev_syn0_wt = syn0_wt
		prev_syn1_wt = syn1_wt

	print ("Synapse 0")
	print (syn0)
	print ("Synapse 0 Update Direction Changes")
	print (syn0_count)
	print ("Synapse 1")
	print (syn1)
	print ("Synapse 1 Update Direction Changes")
	print (syn1_count)