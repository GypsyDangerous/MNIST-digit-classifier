import numpy as np
from perceptron import *
brain = perceptron(784, 2, 256, 10, 900000)
brain.setLearningRate(.0001)


def loadMNIST( prefix, folder ):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( folder + "/" + prefix + '-labels.idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]

    return data, labels

trainingImages, trainingLabels = loadMNIST("train", "C:/Users/david/Downloads/datasets/mnist")
testImages, testLabels = loadMNIST("t10k", "C:/Users/david/Downloads/datasets/mnist")

def toHotEncoding( classification ):
    # emulates the functionality of tf.keras.utils.to_categorical( y )
    hotEncoding = np.zeros([len(classification), 
                              np.max(classification) + 1])
    hotEncoding[np.arange(len(hotEncoding)), classification] = 1
    return hotEncoding

trainingLabels = toHotEncoding(trainingLabels)
testLabels = toHotEncoding(testLabels)

brain.fit(trainingImages, trainingLabels)

brain.test(testImages, testLabels)
