# CS 129.18 Final Project: Face Detection
A face detection system from webcam input that utilizes skin thresholding and canny edge detection to find candidate face locations, and classifies them based on an artificial neural network trained on face datasets. Aside from the ANN, the user can also opt to use the Naive Bayes algorithm for evaluating results.

## Instructions ##
### Using the Main Program ###
To run the program, call `python webcam.py` from the terminal.

### Processing Specific Images ###
Place the images in the `data/input` directory. The second parameter of get_objects_from_file should be True if you want cropped sections of the image, and False if you want the entire annotated image. Run `imgprocess.py`. The resulting images will be in `data/output`.

### Building the Training/Test Sets ###
Put the files you'd like to be part of the training/test sets in the `data/train_images` and `data/test_images` folders, respectively. Call `python gabor.py` from the terminal. The results will be in `data/train.csv` and `data/test.csv`.

### Modifying the Artificial Neural Network ###
The ANN's topology is specified in `mlp.py`'s create_brain function. Running `python jann.py` will either create a new multilayer perceptron (saving the configuration to `mlp.net`) or load the existing one, depending on which is not commented out. To test the ANN's performance on the test set, uncomment the call to load_validation and run the program.

### Testing ###
To test the ANN classification on a pre-existing `data/test.csv`, call `python jann.py`. To test the Naive Bayes classification on a pre-existing `data/test.csv` with its predefined positive and negative means and variances in the main directory, call `python nb.py`.

## Directory Structure ##
```
├── data/
│   ├── train_images/ : 450 images each
│   │   ├── positives/
│   │   ├── negatives/
│   ├── test_images/ : 50 images each
│   │   ├── positives/
│   │   ├── negatives/
│   ├── train.csv
│   ├── test.csv
│   ├── negative_mean.txt
│   ├── positive_mean.txt
│   ├── variance_pos.txt
│   ├── variance_neg.txt
│
├── webcam.py : Main driver file
├── gabor.py : Feature generation, create test/train CSVs
├── imgprocess.py : Process images to detect faces
│
├── jann.py : Artificial Neural Network
├── mlp.py
├── layer.py
├── neuron.py
├── mlp.net : Saved ANN configuration
│
├── nb.py : Naive Bayes
```

## Implementation ##
### Detection ###
To find regions of interest, we use the OpenCV `findContours` function after performing skin thresholding and edge detection.

### Feature Set ###
For each image, the feature vectors are built by computing the mean amplitude and local energies of the response matrices using various convolutions from the Gabor filter.

We generate 16 Gabor filters of four orientations and four frequencies (by changing the theta and lambda parameters of the `getGaborKernel`, respectively). Each convolution results in a matrix, from which the local energy and mean amplitude is computed for a total of 32 features. 

### Artificial Neural Network ###
#### `mlp.net` Format ####

```
[Overall Net Error Upon Saving]
[Topology]
For each layer: 
[Output Weights of each Neuron, including Bias Neuron]
```

## References ##
Aside from the OpenCV docs, we referred to/used the following:

### Gabor Tutorials ###
- http://stackoverflow.com/questions/20608458/gabor-feature-extraction
- http://corpocrat.com/2015/03/25/applying-gabor-filter-on-faces-using-opencv/
- https://skydrive.live.com/redir?resid=8522988C417C6CDA!513

### Datasets ###
The training and test sets used were subsets of the following datasets:
- Caltech Vision, Faces 1999: http://www.vision.caltech.edu/html-files/archive.html
- University of Essex, faces95: http://cswww.essex.ac.uk/mv/allfaces/index.html

### Others ###
- Skin Thresholding: http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
- Automatic Canny Thresholds: http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
