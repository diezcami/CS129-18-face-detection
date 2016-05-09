import numpy as np
import cv2
import os
import csv

POSITIVE_TRAIN_DIR = 'data/train_images/positives'
NEGATIVE_TRAIN_DIR = 'data/train_images/negatives'
POSITIVE_TEST_DIR = 'data/test_images/positives'
NEGATIVE_TEST_DIR = 'data/test_images/negatives'

EPS = 0.00000000000000001

def build_filters():
    filters = []

    # Size of Gabor kernel
    ksize = 31

    # For different orientations
    for theta in np.arange(0, np.pi, np.pi/4):
        # And different wavelengths of the sinusoidal factor
        for lamb in np.arange(np.pi/4, np.pi, np.pi/4):
            # Get a filter
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamb, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)

    return filters
 
# Given an image and a set of filters, derive the response matrices 
def process(img, filters):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100,100))

    responses = []
    # accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        responses.append(fimg)
        # np.maximum(accum, fimg, accum)

    return responses

# Given a response matrix, compute for the local energy
# Local Energy = summing up the squared value of each matrix value from a response matrix
def get_local_energy (matrix):
    local_energy = 0.0
    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            val = int(matrix[row][col]) * int(matrix[row][col])
            local_energy = local_energy + val

    # Divide by the highest possible value, which is 255^2 * (100 x 100)
    # to normalize values from 0 to 1, and replace 0s with EPS value to work with NB
    local_energy = local_energy / 650250000
    return EPS if local_energy == 0 else local_energy

# Given a response matrix, compute for the mean amplitude
# Mean Amplitude = sum of absolute values of each matrix value from a response matrix
def get_mean_amplitude (matrix):
    mean_amp = 0.0

    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            val = abs(int(matrix[row][col]))
            mean_amp = mean_amp + val

    # Divide by the highest possible value, which is 255 * (100 x 100)
    # to normalize values from 0 to 1, and replace 0s with EPS value to work with NB
    mean_amp = mean_amp / 2550000
    return EPS if mean_amp == 0 else mean_amp

# Return a list of images from the given directory
def load_images_from_folder (folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Get the feature vector (local energy/mean amplitude from response matrices) of an image
# This function is called when bulding the CSV files or when processing each frame
def get_image_feature_vector(image, filters, positive=None):
    response_matrices = process(image, filters)

    local_energy_results = []
    mean_amplitude_results = []

    for matrix in response_matrices:
        local_energy = get_local_energy(matrix)
        mean_amplitude = get_mean_amplitude(matrix)
        local_energy_results.append (local_energy)
        mean_amplitude_results.append(mean_amplitude)
        
    if positive is None:
        feature_set = local_energy_results + mean_amplitude_results
    else:
        pos_num = 1 if positive else 0
        feature_set = local_energy_results + mean_amplitude_results + [pos_num]

    return feature_set

# Get all feature vectors of a set of images, used when building the CSV files
def get_all_image_feature_vectors(images, positive):
    filters = build_filters()
    feature_sets = []
    
    for image in images:
        feature_set = get_image_feature_vector(image, filters, positive)
        feature_sets.append (feature_set)

    return feature_sets

# Generates a CSV file containing the feature vectors
def create_csv_output(filename, posdir, negdir):
    # Load images
    positive_images = load_images_from_folder (posdir)
    negative_images = load_images_from_folder (negdir)

    # Get feature vectors of images
    positive_feature_vectors = get_all_image_feature_vectors (positive_images, True)
    negative_feature_vectors = get_all_image_feature_vectors (negative_images, False)

    # Input feature vectors into a CSV file
    with open (filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(positive_feature_vectors)
        writer.writerows(negative_feature_vectors)

if __name__ == '__main__':
    create_csv_output("data/train.csv", POSITIVE_TRAIN_DIR, NEGATIVE_TRAIN_DIR)
    create_csv_output("data/test.csv", POSITIVE_TEST_DIR, NEGATIVE_TEST_DIR)