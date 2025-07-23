
# importing needed libraries in keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils

# importing needed libraries for image pre-processing
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

# importing needed libraries from SKlearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# connecting to google drive (for Colab only)
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/')
os.chdir('/content/drive/MyDrive/Both')

path1 = '/content/drive/MyDrive/Both'        # address where original data are saved
path2 = '/content/drive/MyDrive/reshaped'    # address where reshaped data are saved

img_row, img_col = 224, 224

# determining the data set
imlist = os.listdir(path2)
data = [cv2.imread(path2 + '/' + file) for file in imlist]
data = np.stack(data)  # array of shape [num_images, height, width, channel]

# Initialize one-hot encoded labels
y = np.zeros([1873, 2])
for ii in range(939):
    y[ii, 0] = 1
for ii in range(934):
    y[ii + 939, 1] = 1

x = np.array(data)
y = np.array(y)

(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=1234)

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation

def vgg16_model(img_row, img_col, channel=3, num_classes=2):
    model = VGG16(weights='imagenet', include_top=True)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    x = Dense(2, activation='sigmoid')(model.output)
    model = Model(model.input, x)

    # Set the first 6 layers to non-trainable
    for layer in model.layers[:6]:
        layer.trainable = False

    # Compile model
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Train model
(X_train, X_valid, Y_train, Y_valid) = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=1234)
img_row, img_col = 224, 224
channel = 3
num_classes = 2
batch_size = 32
nb_epoch = 10

model = vgg16_model(224, 224, channel, num_classes)
model.summary()

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid))
model.evaluate(X_test, Y_test)
