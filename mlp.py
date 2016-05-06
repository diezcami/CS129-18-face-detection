from layer import *
from neuron import *

class MLP:
	def __init__(self, topology=None):
		if topology is None:
			self.loadNetwork('mlp.net')
		else:
			self.topology = topology
			self.overallNetError = 0
			self.mLayers = []

			for i,layerNum in enumerate(self.topology):
				newLayer = Layer()
				for neuronNum in range(layerNum):
					if i == 0:
						n = Neuron(0, neuronNum, True, False, False, topology[i+1])
					elif i == len(self.topology)-1:
						n = Neuron(0, neuronNum, False, True, False, 0)
					else:
						n = Neuron(0, neuronNum, False, False, True, topology[i+1])

					newLayer.mNeurons.append(n)

				# Create bias neuron
				if i==0:
					n = Neuron(1, layerNum, True, False, False, topology[i+1])
					n.setBias(True)
					newLayer.mNeurons.append(n)
				elif i != len(self.topology)-1:
					n = Neuron(1, layerNum, False, False, True, topology[i+1])
					n.setBias(True)
					newLayer.mNeurons.append(n)

				self.mLayers.append(newLayer)

	def loadNetwork(self, filename):
		file_obj = open(filename, 'r')

		# Load the overall net error
		self.overallNetError = float(file_obj.readline())

		# Load topology
		self.topology = file_obj.readline().split()
		self.topology = [int(x) for x in self.topology]

		# For each layer...
		self.mLayers = []
		for i,layerNum in enumerate(self.topology):
			newLayer = Layer()

			for neuronNum in range(layerNum):
				outputWeights = file_obj.readline().split()
				outputWeights = [float(ow) for ow in outputWeights]

				if i==0:
					n = Neuron(0, neuronNum, True, False, False, self.topology[i+1], outputWeights)
				elif i == len(self.topology)-1:
					n = Neuron(0, neuronNum, False, True, False, 0, outputWeights)
				else:
					n = Neuron(0, neuronNum, False, False, True, self.topology[i+1], outputWeights)
				newLayer.mNeurons.append(n)

			# Create bias neuron
			if i==0:
				outputWeights = file_obj.readline().split()
				outputWeights = [float(ow) for ow in outputWeights]

				n = Neuron(1, layerNum, True, False, False, self.topology[i+1], outputWeights)
				n.setBias(True)
				newLayer.mNeurons.append(n)
			elif i != len(self.topology)-1:
				outputWeights = file_obj.readline().split()
				outputWeights = [float(ow) for ow in outputWeights]
				
				n = Neuron(1, layerNum, False, False, True, self.topology[i+1], outputWeights)
				n.setBias(True)
				newLayer.mNeurons.append(n)

			self.mLayers.append(newLayer)

		file_obj.close()

	def backPropagate(self, targetValues):
		outputLayer = self.mLayers[-1]
		outputLayerNeurons = outputLayer.mNeurons

		self.overallNetError = 0
		errDelta = [(targetValues[i] - outputLayerNeurons[i].val)**2 for i in range(len(outputLayerNeurons))]
		self.overallNetError = sum(errDelta)

		self.overallNetError = math.sqrt(self.overallNetError / len(outputLayerNeurons))

		# Calculate output neurons' gradients
		for i,n in enumerate(outputLayerNeurons):
			n.calculateOutputGradient(targetValues[i])
			outputLayer.mNeurons[i] = n

		# Calculate gradients on hidden layers
		for i in range(2, len(self.mLayers)+1):
			hiddenLayer = self.mLayers[-i]
			nextLayer = self.mLayers[-i+1]
			hiddenNeurons = hiddenLayer.mNeurons

			for j,n in enumerate(hiddenNeurons):
				n.calculateHiddenGradient(nextLayer.mNeurons)
				hiddenNeurons[j] = n

			# Again, not sure how refs work but this shouldn't be necessary kung pass by reference iyong earlier assignment??
			hiddenLayer.mNeurons = hiddenNeurons

		# Update weights of connections in output and hidden layers
		for i in range(1, len(self.mLayers)):
			layer = self.mLayers[-i]
			prevLayer = self.mLayers[-i-1]
			currentNeurons = layer.mNeurons[:-1]
			prevLayerNeurons = prevLayer.mNeurons

			for n in currentNeurons:
				if i<len(self.mLayers)-1:
					tempBiasNode = prevLayerNeurons[-1]
					prevLayerNeurons = n.updateInputWeights(prevLayerNeurons[:-1])
					prevLayerNeurons.append(tempBiasNode)
				else:
					prevLayerNeurons = n.updateInputWeights(prevLayerNeurons)

				# Again, ff seems unnecessary
				prevLayer.mNeurons = prevLayerNeurons
				self.mLayers[-i-1]=prevLayer


	def feedForward(self, inputs):
		# Step 1: Latch inputs as vals for the input neurons
		inputLayer = self.mLayers[0]
		inputNeurons = inputLayer.mNeurons
		for i,n in enumerate(inputNeurons[:-1]):
			n.val = inputs[i]
			inputNeurons[i] = n # Unnecessary??

		# unnec
		inputLayer.mNeurons = inputNeurons
		self.mLayers[0] = inputLayer

		# Step 2: For each succeeding layer, compute new vals of the neurons
		for i in range(1,len(self.mLayers)):
			currLayer = self.mLayers[i]
			prevLayer = self.mLayers[i-1]

			currNeurons = currLayer.mNeurons
			newNeurons = []

			prevInputs = prevLayer.getInputs()
			for j,currNeuron in enumerate(currNeurons):
				if not currNeuron.getBias():
					prevWeights = prevLayer.getWeights(j)
				else:
					prevWeights = []

				currNeuron.feedForward(prevInputs, prevWeights)
				newNeurons.append(currNeuron)

			currLayer.mNeurons = newNeurons
			self.mLayers[i]=currLayer #unnec

	def printNetwork(self):
		for i, layer in enumerate(self.topology):
			print "==========="
			print "Layer:", i
			print "==========="

			mNeurons = self.mLayers[i].mNeurons
			for j in range(len(mNeurons)):
				n = mNeurons[j]
				print "Neuron:", n.ident, "Value:", n.val
				n.printWeights()

			print ""

	def saveNetwork(self):
		file_obj = open('mlp.net', 'w')

		# Save overall net error
		file_obj.write(str(self.overallNetError) + '\n')

		# Save the topology
		top = ' '.join(str(x) for x in self.topology)
		file_obj.write(top + '\n')

		# For each layer in the topology...
		for i, layer in enumerate(self.topology):
			mNeurons = self.mLayers[i].mNeurons

			# Save the weights for each neuron
			for n in mNeurons: file_obj.write(n.getWeights())

		file_obj.close()
	
	def getInputNeurons(self):
		return self.mLayers[0].mNeurons

	def getOutputNeurons(self):
		return self.mLayers[-1].mNeurons