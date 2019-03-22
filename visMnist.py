import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib import style

# load the MNIST data from files
def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype = 'float32').reshape([ nImages, width, height ])

    labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]

    return data, labels


# put in the file path of your MNIST files
trainingImages, trainingLabels = loadMNIST("train", "C:/Users/david/Downloads/datasets/mnist")
testImages, testLabels = loadMNIST("t10k", "C:/Users/david/Downloads/datasets/mnist")


# convert the labels to hotencoding
def toHotEncoding(classification):
    # emulates the functionality of tf.keras.utils.to_categorical( y )
    hotEncoding = np.zeros([len(classification), 
                              np.max(classification) + 1])
    hotEncoding[np.arange(len(hotEncoding)), classification] = 1
    return hotEncoding







trainingLabels = toHotEncoding(trainingLabels)
testLabels = toHotEncoding(testLabels)
for i in range(100):
	index = np.random.randint(len(testImages))
	img = np.array(testImages[i], dtype=np.float32)
	plt.title("index %d" %(i))
	imgshow = plt.imshow(img, cmap='gray')
	print(testLabels[i])
	plt.show(True)