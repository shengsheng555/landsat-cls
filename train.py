# -*- coding: utf-8 -*-

import os, sys, time, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gdrive_dir = './'
# for PATH in ['./L8_NLCD/']:
#     os.makedirs(PATH, exist_ok=True)
# shutil.copyfile('./drive/My Drive/data/L8_NLCD_extracted_dataset.npy', 
#                  './L8_NLCD/L8_NLCD_extracted_dataset.npy')
arr = np.load('./L8_NLCD_extracted_dataset_blast.npy')

# See the distri
arr_nlcd = arr[:,8,:,:]
freq = np.unique(arr_nlcd, return_counts=True)

leg_num = np.array([0,11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95])
leg_str = np.array(['NaN','Open Water','Perennial Ice/Snow','Developed, Open Space','Developed, Low Intensity',
           'Developed, Medium Intensity','Developed High Intensity','Barren Land (Rock/Sand/Clay)',
           'Deciduous Forest','Evergreen Forest','Mixed Forest','Dwarf Scrub','Shrub/Scrub',
           'Grassland/Herbaceous','Sedge/Herbaceous','Lichens','Moss','Pasture/Hay','Cultivated Crops',
           'Woody Wetlands','Emergent Herbaceous Wetlands'])

# Conversion between original class and training labels
nlcd_dic = {}

# 21 classes, including NaN
nlcd_dic['original'] = {0:20,20:0,11:11,12:12,
                        21:1,1:21,22:2,2:22,23:3,3:23,24:4,4:24,31:5,5:31,41:6,6:41,42:7,7:41,
                        43:8,8:43,51:9,9:51,52:10,10:52,71:13,13:71,72:14,14:72,73:15,15:73,
                        74:16,16:74,81:17,17:81,82:18,18:81,90:19,19:90,95:0,0:95}

# 9 classes, including NaN
nlcd_dic['simple'] = {0:8,11:0,12:0,
                      21:1,22:1,23:1,24:1,31:2,41:3,42:3,
                      43:3,51:4,52:4,71:5,72:5,73:5,
                      74:5,81:6,82:6,90:7,95:7}

# 14 classes, including NaN
nlcd_dic['reduction'] = {0:13,11:0,12:0,
                         21:1,22:2,23:3,24:4,31:5,41:6,42:6,
                         43:6,51:7,52:7,71:8,72:9,73:9,
                         74:9,81:8,82:10,90:11,95:12}

nlcd_dic['reduction_v1'] = {0:13,11:0,12:0,
                            21:1,22:1,23:1,24:1,31:5,41:6,42:6,
                            43:6,51:7,52:7,71:8,72:9,73:9,
                            74:9,81:8,82:10,90:11,95:12}

nlcd_dic['developed'] = {21:0,22:1,23:2,24:3}

sublist_labels = [21, 22, 23, 24]
sublist_labels = [0,11,12,31,41,42,43,51,52,71,72,73,74,81,82,90,95]

def get_patches(arr_l8, arr_nlcd, patch_size=5, mode='reduction', sublist=False):
    '''
    Get patches from numpy 3d or 4d array (multiband images)
    
    Band 8 stores labels 
    Each patch has patch_size*patch_size many pixels. Patch size is odd.
    '''
    s = patch_size
    patches = []
    targets = []
    if len(arr_l8.shape) == 4:
        '''
        Input has shape (N, X, Y, B): (sample_index, band, x-coor, y-coor)
        features has shape (M, patch_size, patch_size, B): (sample_index, band, x-coor, y-coor)
        '''
        N, X, Y, B = arr_l8.shape[0], arr_l8.shape[1], arr_l8.shape[2], arr_l8.shape[3]
        # Pixel (x, y) in sample n is the center of patches[m]
        # m= n*(X-s+1)*(Y-s+1) + (y-2)*(X-s+1) + (x-2), x,y,n starts from 0
        for n in range(N):
            for y in range(Y-s+1):         
                for x in range(X-s+1):
                    if not sublist or arr_nlcd[n, x+s//2, y+s//2] in sublist_labels:
                        # patches.append(arrin[n, :8, x:x+s, y:y+s])
                        # targets.append(nlcd_dic[mode][arrin[n, 8, x+s//2, y+s//2]])
                        patches.append(arr_l8[n, x:x+s, y:y+s, :])
                        targets.append(nlcd_dic[mode][arr_nlcd[n, x+s//2, y+s//2]])

    if len(arr_l8.shape) == 3:
        '''
        Input has shape (B, X, Y): (band, x-coor, y-coor)
        features has shape (M, B, patch_size, patch_size): (sample_index, band, x-coor, y-coor)
        '''  
        X, Y, B = arr_l8.shape[0], arr_l8.shape[1], arr_l8.shape[2]
        # Pixel (x, y) is the center of patches[m], m=(y-2)*(X-s+1)+(x-2), x,y starts from 0
        for y in range(Y-s+1):         
            for x in range(X-s+1):
                if not sublist or arr_nlcd[x+s//2, y+s//2] in sublist_labels:
                    patches.append(arr_l8[x:x+s, y:y+s, :])
                    targets.append(nlcd_dic[mode][arr_nlcd[x+s//2, y+s//2]])

    features = np.array(patches)
    targets = np.array(targets)
    return features, targets

def map_nlcd(arr_nlcd, mode='reduction'):
    for x0 in range(arr3d.shape[0]):
        for x1 in range(arr3d.shape[1]):
            for x2 in range(arr3d.shape[2]):
                arr3d[x0,x1,x2] = nlcd_dic[mode][arr3d[x0,x1,x2]]
    return arr3d            

#########################################
# Check outputs are well ordered
# n = 3
# x = 2
# y = 4
# s = 5
# X = 128
# Y = 128
# m= n*(X-s+1)*(Y-s+1) + (y-2)*(X-s+1) + (x-2)

# feat[m, :, 2, 2] - arr[n, :8, x, y]
# arr[n, 8, x, y] - targ[m]


transform_01 = False
# [0,1] interval for features
arr_nlcd = arr[:,:,:,8].copy()
arr_l8 = arr[:,:,:,:8].copy()
if transform_01:
    arr_l8 = arr_l8.astype('float32')/arr_l8.ravel().max()

from sklearn.model_selection  import train_test_split

# prep = 'seg'
patch_size = 15
prep = 'patch'

# mode = 'simple'
# prep = 'seg'
if prep == 'patch':
    x, y = get_patches(arr_l8, arr_nlcd, 
                       patch_size=patch_size,
                       mode='simple',
                       sublist=False)
else:
    x, y = arr_l8, map_nlcd(arr_l8)

# Randomly split train/valid data using sklearn
x, x_valid, y, y_valid = train_test_split(x, y, test_size=0.1, random_state=0)
x_train = x
y_train = y
freq = np.unique(y_train, return_counts=True)

# freq
# x_valid.shape
num_train = x_train.shape[0]
num_valid = x_valid.shape[0]

"""Training"""

model_id = 'exp0.32'

import os, shutil
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, Dropout
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

patch_size = 15

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

'''CallBacks'''
logdir = ('./' + model_id + '-log')
# logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
for PATH in [logdir]:
    os.makedirs(PATH, exist_ok=True)

from distutils.dir_util import copy_tree
class CustomCallback(tf.keras.callbacks.Callback):
    '''copy log to gdrive'''
    def on_epoch_end(self, epoch, logs=None):
        copy_freq = 1
        if epoch % copy_freq == 0 and epoch > 0:
            print('copy to gdrive...')
            copy_tree(logdir, os.path.join(gdrive_dir, model_id+'-log'))
            # shutil.copytree(logdir, os.path.join(gdrive_dir, model_id+'-log'))  # bad with old python 
            
def get_callbacks(logdir):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(logdir, model_id+'.hdf5'),
        # filepath=os.path.join(logdir, model_id'.epoch{epoch:02d}-val_loss{val_loss:.2f}.hdf5'),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(logdir,'log.csv'), append=True)
    customCallback = CustomCallback()
    return [model_checkpoint_callback, csv_logger
            , customCallback
            ]
callbacks = get_callbacks(logdir)

n_classes = 13
inputShape = (patch_size,patch_size, 8)

# if x_train.shape[1] == 8 and x_valid.shape[1] == 8:  
#     x_train = np.swapaxes(x_train, 1, 3)
#     x_valid = np.swapaxes(x_valid, 1, 3)
y_train_categ = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
y_valid_categ = tf.keras.utils.to_categorical(y_valid, num_classes=n_classes)

inputShape = (patch_size, patch_size, 8)

with strategy.scope():
    model = Sequential()

    l1 = tf.keras.regularizers.l1(0)
    dropout_rate = 0.1
    model.add(tf.keras.layers.BatchNormalization(input_shape=inputShape))
    model.add(Conv2D(8,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                    activity_regularizer=l1))

    # model.add(Conv2D(8,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                    #  activity_regularizer=l1, data_format='channels_first', input_shape=inputShape))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(16,kernel_size=(3, 3), strides=(1, 1), padding='same', 
                    activity_regularizer=l1, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(32,kernel_size=(3, 3), strides=(1, 1), padding='same', 
                    activity_regularizer=l1, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64,kernel_size=(3, 3), strides=(1, 1), padding='same', 
                    activity_regularizer=l1, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(128,kernel_size=(3, 3), strides=(1, 1), padding='same', 
                    activity_regularizer=l1, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(3200, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    # Load model
    # shutil.copyfile(os.path.join(os.path.join(gdrive_dir, model_id+'-log'), model_id+'.hdf5'),
    #                 model_id+'.hdf5')
    # model = tf.keras.models.load_model(model_id+'.hdf5')

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=Adam(lr=0.0003),
                  metrics=['accuracy'] )

model.summary()    
epochs = 40
hs = model.fit(x_train,y_train_categ, batch_size=1024,epochs=epochs, 
                verbose=2,validation_data=(x_valid,y_valid_categ),
                callbacks=callbacks,
                )
