import random
import math

class Weight:
	def __init__(self, val, delta):
		self.val = val
		self.delta = delta

class Neuron:
	def __init__(self, val, ident, isInput, isOutput, isHidden, numOutputs, outputWeights=None):
		self.val = val
		self.ident = ident
		self.isInput = isInput
		self.isOutput = isOutput
		self.isHidden = isHidden
		self.numOutputs = numOutputs
		self.gradient = 0
		self.bias = False

		if outputWeights is None:
			self.outputWeights = [Weight(self.rand(-1,1), 0) for i in range(numOutputs)]
		else:
			self.outputWeights = [Weight(float(outputWeights[i]), 0) for i in range(numOutputs)]
	
	def getBias(self):
		return self.bias

	def setBias(self, bias):
		self.bias = bias

	def updateInputWeights(self, previousLayerNeurons):
		eta = 0.15 # Overall learning rate
		alpha = 0.5 # Momentum

		for i,n in enumerate(previousLayerNeurons):
			oldDeltaWeight = n.outputWeights[self.ident].delta
			newDeltaWeight = 0.15 * n.val * self.gradient + 0.5 * oldDeltaWeight

			w = n.outputWeights[self.ident]
			w.delta = newDeltaWeight
			w.val = w.val + newDeltaWeight # ???

			# Not sure if ff lines are necessary, di ba dapat kung reference naman ito ay naupdate na anyway?
			n.outputWeights[self.ident] = w
			previousLayerNeurons[i] = n

		return previousLayerNeurons

	def feedForward(self, inputs, weights):
		vals = [inp * w for inp, w in zip(inputs, weights)]
		self.val = self.activate(sum(vals))

	def activate(self, val):
		if self.bias: return 1
		else: return math.tanh(float(val))

	def calculateOutputGradient(self, targetVal):
		delta = targetVal - self.val
		self.gradient = delta * self.tanhDerivative(self.val)

	def calculateHiddenGradient(self, neurons):
		dow = self.sumDOW(neurons)
		self.gradient = dow * self.tanhDerivative(self.val)

	def sumDOW(self, neurons):
		vals = [ow.val * n.gradient for ow, n in zip(self.outputWeights, neurons)]
		return sum(vals)

	def tanhDerivative(self, val):
		return 1.0 - self.val * self.val

	def rand(self, mn, mx):
		return random.uniform(mn, mx)

	def printWeights(self):
		for i, ow in enumerate(self.outputWeights):
			print "[", i, "]:", ow.val

	def getWeights(self):
		weights = ' '.join(str(ow.val) for ow in self.outputWeights)
		return weights + '\n'

