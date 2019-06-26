# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages

import keras
from keras.models import Sequential #creates layers of ANN
from keras.layers import Dense #helps edit and customize layers
from keras.layers import Convolution2D #first step of CNN, convolution layers
from keras.layers import MaxPooling2D #Pooling
from keras.layers import Flatten #Converting feature maps into column of entry into ANN

#must prepare dataset images to training and test set are in different folders, 
#and within folders, there are more folders with each category of image (ex. dogs, cats)


#Initializing CNN

classifier = Sequential()

#Step 1, convolution layer

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) #Tensorflow backend
#adding convolution layer (32 feature detectors each 3x3 pixels)
#input_shape refers to number of arguements of image (RGB = 3 or BW = 2) and pixel dimensions (64 x 64)
#always use rectifier for activation function for convolution feature map (increaes non-linearity)


#Step 2 Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))
#size of pool maps to reduce features in ANN

#add another convolutional layer

classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) #don;t need to specify input as keras knows the last layer was a pool
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#Step 3 Flattening
#all pool map values go into one long vector entering ANN

classifier.add(Flatten())
#automatically flattens previous layer


#Making the classic ANN

classifier.add(Dense(output_dim = 128, activation = 'relu')) #hidden layer

classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #output layer

#Compiling CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#specifying optimizer alogrithm (adam means gradient descent, loss function and metrics)


#Fitting CNN to images

#Preprocessing images and doing image augmentation

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) #augmentation of training images to enrich data plus feature scaling

test_datagen = ImageDataGenerator(rescale=1./255) #feature scaling test 

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), #size of images
        batch_size=32, #batch size of descent
        class_mode='binary') #number of categories, binary means 2

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25, #how many times to cycle through training set
        validation_data= test_set,
        validation_steps=2000)



























