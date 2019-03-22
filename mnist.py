import numpy as np
from NeuralNetwork import *
from mnistdata import *
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

brain = perceptron(784, 2, 256, 10, 4)
brain.setLearningRate(.0001)

# put in the file path of your MNIST files
trainingImages, trainingLabels = loadMNIST("train", "C:/Users/david/Downloads/datasets/mnist")
testImages, testLabels = loadMNIST("t10k", "C:/Users/david/Downloads/datasets/mnist")

trainingLabels = toHotEncoding(trainingLabels)
testLabels = toHotEncoding(testLabels)

print("training...")

# train and test the network
brain.fit(trainingImages, trainingLabels)
brain.test(testImages, testLabels)

print(brain.getLearningRate())
for i in range(1): # len(testImages)):
    index = len(testImages)-18 # np.random.randint(len(testImages))
    img = np.array(testImages[index])
    numberlabel = largest_index(testLabels[index])
    plt.title("index %d, label %g" %(index, numberlabel))
    imgshow = plt.imshow(img, cmap='gray')
    plt.show(True)
