import numpy as np
import cv2
import os
import csv
 
INPUT_DIR = 'input/'

def build_filters():
    filters = []
    ksize = 31

    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize: size of gabor kernell
    # sigma: standard dev of gaussian function
    # theta: orientation/angle (between 0-pi)
    # lambda:  wavelength of the sinusoidal factor
    # gamma: spatial aspect ratio
    # psi: phase offset
    # ktype: type and range of values that each pixel in the gabor kernel can hold
    for theta in np.arange(0, np.pi, np.pi / 4):
        for lamb in np.arange(0, np.pi, np.pi / 4):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamb, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)

    return filters
 
def process(img, filters):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
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
    local_energy = 0
    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            local_energy = local_energy + int(matrix[row][col]) * int(matrix[row][col])
    return local_energy

# Given a response matrix, compute for the mean amplitude
# Mean Amplitude = sum of absolute values of each matrix value from a response matrix
def get_mean_amplitude (matrix):
    mean_amp = 0
    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            mean_amp = mean_amp + abs(int(matrix[row][col]))
    return mean_amp

def load_images_from_folder (folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

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
        feature_set = local_energy_results + mean_amplitude_results + [positive]

    return feature_set

def get_all_image_feature_vectors(images, positive):
    filters = build_filters()
    feature_sets = []
    
    for image in images:
        feature_set = get_image_feature_vector(image, filters, positive)
        feature_sets.append (feature_set)

    return feature_sets

# Generates a CSV file containing the feature vectors
def create_csv_output():
    # Load images
    positive_images = load_images_from_folder (INPUT_DIR)
    # negative_images = load_images_from_folder (NEGATIVE_DIR)

    # Get feature vectors of images
    positive_feature_vectors = get_all_image_feature_vectors (positive_images, True)
    # negative_feature_vectors = get_all_image_feature_vectors (negative_images, False)

    # Input feature vectors into a CSV file
    with open ("data/training_data/train.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(positive_feature_vectors)
        # writer.writerows(negative_feature_vectors)

if __name__ == '__main__':
    create_csv_output()
    

