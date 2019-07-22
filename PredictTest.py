import os
import numpy as np
from keras import optimizers
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Reshape, TimeDistributed, Bidirectional,
                          GRU, BatchNormalization)
from keras.models import Sequential
from keras.models import load_model
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

# Directories
dirpath = os.getcwd()
train_dir = os.path.join(dirpath,'data_norm/train')
test_dir = os.path.join(dirpath,'data_norm/test')

# Define training and validation generators
generator = VideoGenerator(train_dir=train_dir,
                           test_dir=test_dir,
                           dims=(FRAME_CAP, MOUTH_WIDTH, MOUTH_HEIGHT, NUM_CHANNELS),
                           batch_size=1,
                           horizontal_flip=True,
                           shuffle=False,
                           file_ext=".np*")

testing_generator = generator.generate(train_or_test="test")
testing_steps_per_epoch = len(generator.filenames_test)
print(len(generator.filenames_test))


label_map = generator.classname_by_id
print('label_map', label_map)


# load model
model = load_model('model_train.h5')
# summarize model.
model.summary()

#scores = model.evaluate_generator(generator=testing_generator, steps=testing_steps_per_epoch)
#print(model.metrics_names, scores)
# Test data predictions
predictions = model.predict_generator(generator=testing_generator, steps=testing_steps_per_epoch)
print('predictions before ', predictions.shape)
predictions = np.argmax(predictions, axis=-1) #multiple categories
print('predictions', predictions.shape)
#print('shuffle_filenames', len(generator.shuffle_filenames))

# class_indices: mapping from class name to class indices
label_map = generator.classname_by_id
print('label_map', label_map)
predictions = [label_map[k] for k in predictions]
correct = 0
for i, prediction in enumerate(predictions):
    print("%s  %s" % (generator.filenames_test[i],prediction))
    if prediction in generator.filenames_test[i]:
        correct += 1


print('Accuracy ', correct/len(generator.filenames_test))

