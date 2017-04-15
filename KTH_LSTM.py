'''
This program demonstrates how to train and use a deep neural network with
2D Convolutional and recurrent layers, for activity recognition in videos.
I selected the KTH dataset since it is relatively small and has activities
that are easy to learn.

Chamin Morikawa
Last updated 2017-04-15
'''

from __future__ import print_function
import numpy as np

# for file operations
import os
from PIL import Image

import keras.backend as K
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU # you can also try using GRU layers
from keras.optimizers import RMSprop, Adadelta, adam, sgd # you can try all these optimizers
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, TimeDistributedDense, Dropout, Reshape, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from random import randint
import gc
# natural sorting
import re

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

# use this if you want reproducibility
#np.random.seed(2016)

# we will be using tensorflow
K.set_image_dim_ordering('tf')

# specifiy the path to your KTH data folder
trg_data_root = "/path/to/dataset/KTH/"

# load training or validation data
# with 25 persons in the dataset, start_index and finish_index has to be in the range [1..25]
def load_data_for_persons(root_folder, start_index, finish_index, frames_per_clip):
    # these strings are needed for creating subfolder names
    class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"] # 6 labels
    frame_path = "/frames/"
    frame_set_prefix = "person" # 2 digit person ID [01..25] follows
    rec_prefix = "d" # seq ID [1..4] follows
    rec_count = 4
    seg_prefix = "seg" # seq ID [1..4] follows
    seg_count = 4

    data_array = []
    classes_array = []

    # let's make a couple of loops to generate all of them
    for i in range(0, len(class_labels)):
        # class
        class_folder = trg_data_root + class_labels[i] + frame_path

        for j in range(start_index, finish_index+1):
            # person
            if j<10:
                person_folder = class_folder + frame_set_prefix + "0" + str(j) + "_" + class_labels[i] + "_"
            else:
                person_folder = class_folder + frame_set_prefix + str(j) + "_" + class_labels[i] + "_"

            for k in range(1,rec_count+1):
                # recording
                rec_folder = person_folder + rec_prefix + str(k) + "/"
                for m in range(1,seg_count+1):
                    # segment
                    seg_folder = rec_folder + seg_prefix + str(m) + "/"

                    # get the list of files
                    file_list = [f for f in os.listdir(seg_folder)]
                    example_size = len(file_list)

                    # for larger segments, we can change the starting point to augment the data
                    clip_start_index = 0
                    if example_size > frames_per_clip:
                        # set a random starting point but fix length - augments data, but slows training
                        #clip_start_index = randint(0, (example_size - frames_per_clip))
                        # sample the frames from the center
                        clip_start_index = example_size/2 - frames_per_clip/2
                        example_size = frames_per_clip

                    # need natural sort before loading data
                    file_list.sort(key=natural_sort_key)

                    #create a list for each segment
                    current_seg_temp = []
                    for n in range(clip_start_index,example_size+clip_start_index):
                        file_path = seg_folder + file_list[i]
                        data = np.asarray( Image.open( file_path), dtype='uint8' )
                        # remove unnecessary channels
                        data_gray = np.delete(data,[1,2],2)
                        data_gray = data_gray.astype('float32')/255.0
                        current_seg_temp.append(data_gray)

                    # preprocessing
                    current_seg = np.asarray(current_seg_temp)
                    current_seg = current_seg.astype('float32')

                    data_array.append(current_seg)
                    classes_array.append(i)

    # # create one-hot vectors from output values
    classes_one_hot = np.zeros((len(classes_array), len(class_labels)))
    classes_one_hot[np.arange(len(classes_array)), classes_array] = 1

    # done
    return (np.array(data_array), classes_one_hot)

# what you need to know about data, to build the model
img_rows = 120
img_cols = 160
maxToAdd = 25 # use 25 consecutive frames from each video segment, as a training sample
nb_classes = 6
class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

# build network model
print("Building model")

#define our time-distributed setup
model = Sequential()

# note: the architecture below is just for demonstration.
# for higher accuracy, you will have to change the number and dimensions of layers.

# three convolutional layers
model.add(TimeDistributed(Convolution2D(4, 5, 5, subsample=(2, 2), border_mode='valid'), input_shape=(maxToAdd,img_rows,img_cols,1)))
model.add(Activation('relu'))

# not sure why I cannot add pooling layers
#model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

model.add(TimeDistributed(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='valid')))
model.add(Activation('relu'))

model.add(TimeDistributed(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid')))
model.add(Activation('relu'))

# flatten and prepare to go for recurrent learning
model.add(TimeDistributed(Flatten()))

# a single dense layer
model.add(TimeDistributed(Dense(80)))
model.add(BatchNormalization()) # required for ensuring that the network learns
model.add(Activation('relu'))

# GRU layers
#model.add(GRU(output_dim=100,return_sequences=True))
#model.add(GRU(output_dim=50,return_sequences=False))

# the LSTM layer performed better than GRU layers
model.add(LSTM(output_dim =80, activation = 'tanh'))

# let's try some dropping out here
model.add(Dropout(.1))

# fully connected layers to finish off
model.add(Dense(80, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop') #adam is faster, but you can use the others too.

#training parameters
batch_size = 16 # increase if your system can cope with more data
nb_epochs = 12 # I once achieved 77.5% accuracy with 100 epochs. Feel free to change

print ("Loading data")
# load training data
X_train, y_train = load_data_for_persons(trg_data_root, 1, 25, maxToAdd)
# NOTE: if you can't fit all data in memory, load a few users at a time and
# use multiple epochs. I don't recommend using one user at a time, since
# it prevents good shuffling.

# perform training
print("Training")
model.fit(np.array(X_train), y_train, batch_size=batch_size, nb_epoch=nb_epochs, shuffle=True, verbose=1)

# clean up the memory
X_train       = None
y_train       = None
X_val = None
y_val = None
gc.collect()

print("Testing")

# load test data: in this case, person 9
X_test, y_test = load_data_for_persons(trg_data_root, 9, 9, maxToAdd)
print('Total no. of testing samples used:', y_test.shape[0])

preds = model.predict(np.array(X_test))

confusion_matrix = np.zeros(shape=(y_test.shape[1],y_test.shape[1]))
accurate_count = 0.0
for i in range(0,len(preds)):
    # updating confusion matrix
    confusion_matrix[np.argmax(preds[i])][np.argmax(np.array(y_test[i]))] += 1

    # if you are not sure of the axes of the confusion matrix, try the following line
    #print ('Predicted: ', np.argmax(preds[i]), ', actual: ', np.argmax(np.array(y_val_one_hot[i])))

    # calculating overall accuracy
    if np.argmax(preds[i])==np.argmax(np.array(y_test[i])):
        accurate_count += 1

print('Validation accuracy: ', 100*accurate_count/len(preds)),' %'
print('Confusion matrix:')
print(class_labels)
print(confusion_matrix)

#save the model
jsonstring  = model.to_json()
with open("KTH_LSTM.json",'wb') as f:
    f.write(jsonstring)
model.save_weights("KTH_LSTM.h5",overwrite=True)

# done.
