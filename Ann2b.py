from joblib.numpy_pickle_utils import xrange
from numpy import *

import csv
rows = []
with open("featureslist.csv", 'r', csv.QUOTE_NONNUMERIC) as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        #rows.append(row)
        line=[]
        for y in row:
            t = float(y)
            line.append(t)
           # print(t)
        rows.append(line)
#print(header)
#print("---")
print(rows[0])


class NeuralNet(object):
	def __init__(self):
		# Generate random numbers
		random.seed(1)

		# Assign random weights to a 3 x 1 matrix,
		self.synaptic_weights = 2 * random.random((12, 1)) - 1

	# The Sigmoid function
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# Train the neural network and adjust the weights each time.
	def train(self, inputs, outputs, training_iterations):
		for iteration in xrange(training_iterations):
			 
			output = self.learn(inputs)

			 
			error = outputs - output

			 
			#factor = dot(inputs.T, error * self.__sigmoid_derivative(output))
			#self.synaptic_weights += factor

		# The neural network thinks.

	def learn(self, inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))


 
neural_network = NeuralNet()
ot=[[271.5265584,279.4971183,1118.886101,316,237,392,234.4975722,190.4042983,1035.059011,3914.432903,4607.328737,3987.621972]]


#inputs = array([[0,1,1,1,0,0,0,1,1,1,0,0,0], [0,1,1,1,0,0,0,1,1,1,0,0,0], [0,1,1,1,0,0,0,1,1,1,0,0,0]])
#outputs = array(rows[0]).T
inputs = array(rows).T
outputs = array(ot)

neural_network.train(inputs, outputs, 10000)
 
print(neural_network.learn(outputs))

