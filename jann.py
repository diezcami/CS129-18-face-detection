from mlp import MLP
import csv

def load_training(filename, brain):
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			data = [float(x) for x in row[:-1]]
			label = row[-1]

			target = [0]
			target[0] = int(float(label))

			idealError = 0.3
			epoch = 5

			for i in range(epoch):
				currError = brain.overallNetError
				brain.feedForward(data)	

				output = brain.getOutputNeurons()				
				brain.backPropagate(target)

				if brain.overallNetError < currError:
					if brain.overallNetError < idealError:
						break

	return brain

def get_ann_label(data, brain):
	brain.feedForward(data)
	output = brain.getOutputNeurons()
	return 1 if output[0].val > 0 else 0

def load_validation(filename, brain):
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		numCorrect = 0
		numTotal = 0
		numFalsePositive = 0
		numFalseNegative = 0
		numTruePositive = 0
		numTrueNegative = 0

		for row in reader:
			data = [float(x) for x in row[:-1]]
			label = int(float(row[-1]))
			annLabel = get_ann_label(data, brain)

			if annLabel == label:
				numCorrect = numCorrect + 1
			if annLabel == 1 and label == 0:
				numFalsePositive += 1
			if annLabel == 1 and label == 1:
				numTruePositive += 1
			if annLabel == 0 and label == 0:
				numTrueNegative += 1
			if annLabel == 0 and label == 1:
				numFalseNegative += 1
			numTotal = numTotal + 1

		print "ANN Results"
		print "Total Labels Correct:", numCorrect
		print "Total Labels Evaluated: ", numTotal
		print "Accuracy: ", float(numCorrect)/float(numTotal)*100.0, "%"

		print "Raw True Positive: ", numTruePositive
		print "Raw True Negative: ", numTrueNegative
		print "Raw False Positive: ", numFalsePositive
		print "Raw False Negative: ", numFalseNegative

		print "True Positive Rate: ", float(numTruePositive)/(numTruePositive+numFalseNegative)
		print "True Negative Rate: ", float(numTrueNegative)/(numFalsePositive+numTrueNegative)
		print "False Positive Rate: ", float(numFalsePositive)/(numFalsePositive+numTrueNegative)
		print "False Negative Rate: ", float(numFalseNegative)/(numTruePositive+numFalseNegative)

# Create new MLP
def create_brain():
	topology = [24,48,24,12,1]
	brain = MLP(topology)
	brain = load_training('data/train.csv', brain)
	brain.saveNetwork()

	return brain
	
# Load existing topology from mlp.net
def get_brain():
	brain = MLP()
	return brain

if __name__ == '__main__': 
	# If you want to test on a new brain:
	# brain = create_brain()

	# If you want to test on the existing brain:
	brain = get_brain()

	# Run the tests
	load_validation('data/test.csv', brain)