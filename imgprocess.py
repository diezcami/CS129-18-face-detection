import cv2
import numpy as np

from gabor import get_image_feature_vector, build_filters

INPUT_DIR = 'data/input/'
OUTPUT_DIR = 'data/output/'        

def get_objects_from_file(filename):
	# Load image
	orig = cv2.imread(INPUT_DIR + filename)
	get_objects_from_img(orig, filename)

def get_objects_from_img(orig, filename=".jpg", train=False):
	# Convert to gray, equalize to improve contrast
	img = orig.copy()
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	imgray = cv2.equalizeHist(imgray)

	# Perform Canny
	edges = cv2.Canny(imgray,100,300)

	# Find contours
	ret,thresh = cv2.threshold(edges,100,255,0)
	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

	filters = build_filters()

	# Draw contours
	for i in range(len(contours)):
		image = cv2.drawContours(img, contours, i, (0,255,0), 1)
		x,y,w,h = cv2.boundingRect(contours[i])
		
		# Arbitrary threshold height and width is 20, 20:
		if h>50 and w>50: 
			cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)
			cropimg = orig[y:y+h, x:x+w]
			
			if train:
				output_name = OUTPUT_DIR + filename[:-4] + str(i) + '.jpg'
				cv2.imwrite(output_name, cropimg)
			else:
				feature_set = get_image_feature_vector(cropimg, filters)
				# Check via ANN whether given this feature set is a face or not
			
			# To show individual images in a window:
			# cv2.imshow('Cropped Image ' + str(i), cropimg)
		
	# cv2.imshow('Original Image', orig)
	cv2.imshow('Annotated Image', image)
	# cv2.imshow('Contrast Image', imgray)
	# cv2.waitKey(0)

# get_objects_from_file('woman.jpg')