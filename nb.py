# Some constant declarations
POSITIVE_PROBABILITY = 0.398809524
NEGATIVE_PROBABILITY = 0.601190476

# Actual NB algorithm. Returns 1 if the image is a face and 0 otherwise.
def get_nb_label(feature_vector):
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

# Retrieves mean and variance information
# Called in the check_objectivity method
def process_training_data (file_name):
	file_address = 'data/training_data/' + file_name + '.txt'
	training_data = open(file_address,'r')
	dimension_data = []
	for dimension in training_data:
		dimension_data.append(dimension)

	return dimension_data