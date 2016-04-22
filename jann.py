from mlp import MLP
import csv

classes = {
	'A': 0,
	'B': 1,
	'C': 2
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

def loadValidation(filename, brain):
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		numCorrect = 0
		numTotal = 0

		for row in reader:
			data = [float(x) for x in row[:-1]]
			label = row[-1]

			brain.feedForward(data)
			output = brain.getOutputNeurons()
			outputVals = [o.val for o in output]

			for i,o in enumerate(outputVals):
				if o==max(outputVals):
					if label=='A' and i==0:
						numCorrect = numCorrect + 1
					if label=='B' and i==1:
						numCorrect = numCorrect + 1
					if label=='C' and i==2:
						numCorrect = numCorrect + 1
			numTotal = numTotal + 1

		print "Correct:", numCorrect
		print "Total: ", numTotal
		print "Accuracy: ", float(numCorrect)/float(numTotal)*100.0, "%"

def main():
	# If new topology:
	# topology = [3,4,4,3]
	# brain = MLP(topology)
	# loadTraining('train.csv', brain)
	# brain.saveNetwork()

	# If loading existing topology in mlp.net:
	brain = MLP()

	loadValidation('test.csv', brain)

if __name__ == '__main__': main()