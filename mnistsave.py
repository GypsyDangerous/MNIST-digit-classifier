import numpy as np
from NeuralNetwork import *
from mnistdata import *
brain = perceptron(784, 2, 256, 10, 10)
brain.load()
eps = int(input("input epochs: "))
brain.setEpochs(eps) # 10000000
brain.setLearningRate(.00005)

# put in the file path of your MNIST files
trainingImages, trainingLabels = loadMNIST("train", "C:/Users/david/Downloads/datasets/mnist")
testImages, testLabels = loadMNIST("t10k", "C:/Users/david/Downloads/datasets/mnist")

trainingLabels = toHotEncoding(trainingLabels)
testLabels = toHotEncoding(testLabels)

# train and test the network
brain.fit(trainingImages, trainingLabels)

brain.test(testImages, testLabels)
print(brain.getLearningRate())
brain.save()
