from mlp import MLP
import csv

classes = {
	'TRUE': 1,
	'FALSE': 0
}

classesReverse = {
	'1' : 'TRUE',
	'0' : 'FALSE'
}

def loadTraining(filename, brain):
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			data = [float(x) for x in row[:-1]]
			label = row[-1]

			target = [0] * len(classes)
			target[classes[label]] = 1

			idealError = 0.5
			epoch = 3

			for i in range(epoch):
				currError = brain.overallNetError
				brain.feedForward(data)
				brain.backPropagate(target)

				if brain.overallNetError < currError:
					if brain.overallNetError < idealError:
						break

def getLabel(data):
	brain.feedForward(data)
	output = brain.getOutputNeurons()
	outputVals = [o.val for o in output]

	for i,o in enumerate(outputVals):
		if o == max(outputVals):
			return classesReverse[i]

def loadValidation(filename, brain):
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		numCorrect = 0
		numTotal = 0

		for row in reader:
			data = [float(x) for x in row[:-1]]
			label = row[-1]
			annLabel = getLabel(data)

			if annLabel == label:
				numCorrect = numCorrect + 1
			numTotal = numTotal + 1

		print "Correct:", numCorrect
		print "Total: ", numTotal
		print "Accuracy: ", float(numCorrect)/float(numTotal)*100.0, "%"

def ann():
	# If new topology:
	topology = [32,16,32,16,2]
	brain = MLP(topology)
	loadTraining('train.csv', brain)
	brain.saveNetwork()

	# If loading existing topology in mlp.net:
	# brain = MLP()

	# loadValidation('test.csv', brain)

if __name__ == '__main__': main()