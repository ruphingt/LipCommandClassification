"""
To be run under the LipInteract folder
This script will train a 3D-CNN + RNN classification model and save the trained
model as model_train.h5

To be run within the LipInteract folder

"""

import os
import numpy as np
from keras import optimizers
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Reshape, TimeDistributed, Bidirectional,
                          GRU, BatchNormalization)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from VideoGenerator import VideoGenerator

# Input shape
FRAME_CAP = 70
MOUTH_WIDTH = 100
MOUTH_HEIGHT = 80
NUM_CHANNELS = 3
NUM_CLASSES = 10
vid_shape = (FRAME_CAP, MOUTH_HEIGHT, MOUTH_WIDTH, NUM_CHANNELS)

# Number of hidden units in GRU
NUM_HIDDEN_UNITS = 96
learning_rate = 0.001
epochs = 200
decay_rate = learning_rate/ epochs
ADAM_OPTIMIZER = optimizers.Adam(lr=learning_rate, decay=decay_rate)

# Directories
dirpath = os.getcwd()
train_dir = os.path.join(dirpath,'data_norm/train')
test_dir = os.path.join(dirpath,'data_norm/test')

batch_size = 16

# Define training and validation generators
generator = VideoGenerator(train_dir=train_dir,
                           test_dir=test_dir,
                           dims=(FRAME_CAP, MOUTH_WIDTH, MOUTH_HEIGHT, NUM_CHANNELS),
                           batch_size=batch_size,
                           horizontal_flip=True,
                           shuffle=True,
                           file_ext=".np*")

training_generator = generator.generate(train_or_test='train')
training_steps_per_epoch = len(generator.filenames_train) // batch_size
testing_generator = generator.generate(train_or_test="test")
testing_steps_per_epoch = len(generator.filenames_test) // batch_size

# Define model
model = Sequential()

# 3D Convolution Model
model.add(Conv3D(32, kernel_size=(3, 7, 7), strides=(1, 3, 3), padding='same', input_shape=vid_shape))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model.add(BatchNormalization(axis=4))# Do batch normalization on the channels
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 1, 1), padding='same'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid'))
model.add(BatchNormalization(axis=4))# Do batch normalization on the channels
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv3D(96, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid'))
model.add(BatchNormalization(axis=4))# Do batch normalization on the channels
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Flatten the output
#model.add(Reshape((70,1152)))
model.add(TimeDistributed(Flatten()))
#model.add(Flatten())

# Bidrectional GRU
model.add(Bidirectional(GRU(NUM_HIDDEN_UNITS, return_sequences=True)))
model.add(Bidirectional(GRU(NUM_HIDDEN_UNITS)))

# FC Softmax
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=ADAM_OPTIMIZER, metrics=["accuracy"])
model.summary()
model.fit_generator(generator=training_generator,
                    steps_per_epoch=training_steps_per_epoch,
                    epochs=15)

model.summary()

# Evaluate the model
# Scores for training data set
#scores = model.evaluate(Xtrain, train_label, verbose=0)
# Scores for test data set

#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save model and architecture to single file
model.save("model_train_.h5")
print("Saved model to disk")


