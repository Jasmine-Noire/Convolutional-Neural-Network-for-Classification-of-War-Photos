# importing needed libraries in keras
from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Dropout,Activation
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

#importing neede libraries for image pre-processing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import theano
from PIL import Image
from numpy import *
import cv2
import os

#importing needed libraries from SKlearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#connecting to google drive
from google.colab import drive

drive.mount('/content/drive')
%cd /content/drive/MyDrive/

os.chdir('/content/drive/MyDrive/Both')

path1='/content/drive/MyDrive/Both'          #adress where original data are save
path2='/content/drive/MyDrive/reshaped'      #adress where reshaped data are saved


img_row=224
img_col=224
#start preprocessing
#listing=os.listdir(path1)
#for file in listing:
   # im=Image.open(path1 + '/' + file  )
   # img=im.resize((img_row,img_col))
    #img.save(path2 +'/' + file, "JPEG" )


# determining the data set
imlist=os.listdir(path2)

data=[]
for file in imlist:
   data.append(cv2.imread(path2 + '/' + file))

data = np.stack(data) # array of shape [num_images, height, width, channel]

# Initialize ohe_labels as all zeros
import numpy as np
y = np.zeros([1873,2])


# Loop over the labels

for ii in range(939):

    # Find the location of this label in the categories variable

    y[ii,0]=1

    # Set the corresponding zero to one
for ii in  range(934):   

    y[ii+939,1] = 1


x=np.array(data)
y=np.array(y)

(X_train,X_test,Y_train,Y_test)=train_test_split(x,y, shuffle=True, test_size=0.2, random_state=1234)


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss




from keras.models import Model

def vgg16_model(img_row, img_col, channel=3, num_classes=2):

    model = VGG16(weights='imagenet', include_top=True)

    model.layers.pop()

    model.outputs = [model.layers[-1].output]

    #model.layers[-1].outbound_nodes = []

    x=Dense(2, activation='sigmoid')(model.output)

    model=Model(model.input,x)


#To set the first 8 layers to non-trainable (weights will not be updated)

    for layer in model.layers[:6]:

       layer.trainable = False

# Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Example to fine-tune on 3000 samples from Cifar10
(X_train,X_valid,Y_train,Y_valid)=train_test_split(x,y, shuffle=True, test_size=0.2, random_state=1234)
img_row, img_col = 224, 224  # Resolution of inputs
channel = 3
num_classes = 2 
batch_size = 32
nb_epoch = 10

# Load our model
model = vgg16_model(224, 224, channel, num_classes)

model.summary()

# Start Fine-tuning
model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_valid, Y_valid))

model.evaluate(X_test,Y_test)
