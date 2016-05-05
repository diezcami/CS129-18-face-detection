import numpy as np
import cv2
import os
 
INPUT_DIR = 'matrix/'

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
        # ksize: size of gabor kernell
        # sigma: standard dev of gaussian function
        # theta: orientation/angle (between 0-pi)
        # lambda:  wavelength of the sinusoidal factor
        # gamma: spatial aspect ratio
        # psi: phase offset
        # ktype: type and range of values that each pixel in the gabor kernel can hold
        # 
        # (1) I'm not sure how many scales and orientations are needed for the convolution process
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
 
def process(img, filters):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Double precision: Comment back in, doesn't work on my OpenCV
    # img = im2double(img);
    img = cv2.resize(img, (100,100))

    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)

    # (2) I'm not sure how to get the response matrix, but it should probably be somewhere here
    return accum

# Given a response matrix, compute for the local energy
# Local Energy = summing up the squared value of each matrix value from a response matrix
def get_local_energy (matrix):
    local_energy = 0
    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            local_energy = local_energy + matrix[row][col] * matrix[row][col]
    return local_energy

# Given a response matrix, compute for the mean amplitude
# Mean Amplitude = sum of absolute values of each matrix value from a response matrix
def get_mean_amplitude (matrix):
    mean_amp = 0
    for row in range (len(matrix)):
        for col in range(len(matrix[0])):
            mean_amp = mean_amp + abs(matrix[row][col])
    return mean_amp

def load_images_from_folder (folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

if __name__ == '__main__':
    images = load_images_from_folder (INPUT_DIR)
    filters = build_filters()
    feature_sets = []

    for i, image in enumerate(images):
        res1 = process(image, filters)
        response_matrices = []
        # cv2.imshow('result', res1) #Testing purposes
        # Insert code to get response matrices
    
        local_energy_results = []
        mean_amplitude_results = []

        for matrix in response_matrices:
            local_energy = get_local_energy(matrix)
            mean_amplitude = get_mean_amplitude(matrix)
            local_energy_results.append (local_energy)
            mean_amplitude_results.append(mean_amplitude)

        feature_set = local_energy_results + mean_amplitude_results
        feature_sets.append (feature_set)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()