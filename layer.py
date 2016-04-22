class Layer:
	def __init__(self):
		self.mNeurons = []

	def getInputs(self):
		inputs = [n.val for n in self.mNeurons]
		return inputs

	def getWeights(self, c):
		weights = [n.outputWeights[c].val for n in self.mNeurons]
		return weights