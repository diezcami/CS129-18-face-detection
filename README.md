# CS 129.18 Project
An object-detection platform that evaluates results from the Haars classifier algorithm using a neural network and Naive Bayes.

## Using the Programm ##
In the main function of jann.py, comment out line 68 and uncomment lines 62-65 if a new topology should be created, the network trained using train.csv, saved into mlp.net, and tested on test.csv. If an existing topology should be loaded from mlp.net, comment out lines 62-65 and uncomment line 68. Run 'python jann.py'. Results will be printed in standard out.

## mlp.net Format ##
[Overall Net Error Upon Saving]
[Topology]

For each layer: 
[Output Weights of each Neuron, including Bias Neuron]