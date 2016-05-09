import csv, math

# Actual NB algorithm. Returns 1 if the image is a face and 0 otherwise.
def get_nb_label(feature_vector):
	POSITIVE_PROBABILITY = 0.5
	NEGATIVE_PROBABILITY = 0.5
	positive_mean = process_training_data('positive_mean')
	negative_mean = process_training_data('negative_mean')

	variance_pos = process_training_data('variance_pos')
	variance_neg = process_training_data('variance_neg')

	prod_positive = 1.0
	prod_negative = 1.0
	for i in range(len(feature_vector)):
		base_pos = 1 / math.sqrt(2 * math.pi * float(variance_pos[i]))
		base_neg = 1 / math.sqrt(2 * math.pi * float(variance_neg[i]))
		
		positive_mult = math.e ** ((-1 * (float(feature_vector[i]) - float(positive_mean[i])) ** 2) / (2 * float(variance_pos[i])))
		negative_mult = math.e ** ((-1 * (float(feature_vector[i]) - float(negative_mean[i])) ** 2) / (2 * float(variance_neg[i])))
		
		prod_positive = prod_positive * (base_pos * positive_mult)
		prod_negative = prod_negative * (base_neg * negative_mult)
	
	prod_positive = prod_positive * POSITIVE_PROBABILITY;
	prod_negative = prod_negative * NEGATIVE_PROBABILITY;
	
	if prod_positive > prod_negative:
		return 1
	else:
		return 0

def load_validation(filename):
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
			nbLabel = get_nb_label(data)

			if nbLabel == label:
				numCorrect = numCorrect + 1
			if nbLabel == 1 and label == 0:
				numFalsePositive += 1
			if nbLabel == 1 and label == 1:
				numTruePositive += 1
			if nbLabel == 0 and label == 0:
				numTrueNegative += 1
			if nbLabel == 0 and label == 1:
				numFalseNegative += 1
			numTotal = numTotal + 1

		print "NB Results"
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

# Retrieves mean and variance information
# Called in the check_objectivity method
def process_training_data (file_name):
	file_address = 'data/training_data/' + file_name + '.txt'
	training_data = open(file_address,'r')
	dimension_data = []
	for dimension in training_data:
		dimension_data.append(dimension)

	return dimension_data

if __name__ == '__main__': 
	# Run the tests
	load_validation('data/test.csv')