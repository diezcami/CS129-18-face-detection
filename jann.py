from mlp import MLP
import csv

def load_training(filename, brain):
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			data = [float(x) for x in row[:-1]]
			label = row[-1]

			target = [0, 0]
			target[int(label)] = 1

			idealError = 0.3
			epoch = 3

			for i in range(epoch):
				currError = brain.overallNetError
				brain.feedForward(data)
				brain.backPropagate(target)

				if brain.overallNetError < currError:
					if brain.overallNetError < idealError:
						break

def get_ann_label(data, brain):
	brain.feedForward(data)
	output = brain.getOutputNeurons()
	outputVals = [o.val for o in output]

	for i,o in enumerate(outputVals):
		if o == max(outputVals):
			return i

def load_validation(filename, brain):
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		numCorrect = 0
		numTotal = 0

		for row in reader:
			data = [float(x) for x in row[:-1]]
			label = row[-1]
			annLabel = get_ann_label(data)

			if annLabel == label:
				numCorrect = numCorrect + 1
			numTotal = numTotal + 1

		print "Correct:", numCorrect
		print "Total: ", numTotal
		print "Accuracy: ", float(numCorrect)/float(numTotal)*100.0, "%"

# Create new MLP
def create_brain():
	topology = [32,16,32,16,2]
	brain = MLP(topology)
	load_training('data/training_data/train.csv', brain)
	brain.saveNetwork()
	
# Load existing topology from mlp.net
def get_brain():
	brain = MLP()
	return brain

# load_validation('test.csv', brain)

if __name__ == '__main__': create_brain()