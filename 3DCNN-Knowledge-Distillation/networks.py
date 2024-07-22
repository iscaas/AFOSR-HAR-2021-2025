# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:59:09 2022

@author: hayatu
"""
import tensorflow  
from tensorflow.keras.models import Sequential
import keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras import Model
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'

def C3D(weights='sports1M'):
    
    if K.image_data_format() == 'channels_last':
      shape = (16,112,112,3) 
    else:
      shape = (3,16,112,112)

        
    model = Sequential()
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0,1,1)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))
    
    model.add(Flatten())
    
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if weights == 'sports1M':
        weights_path = get_file('sports1M_weights_tf.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='b7a93b2f9156ccbebe3ca24b41fc5402')
        
        model.load_weights(weights_path)


    return model



def Teacher_C3D(num_of_classes):
    
    if K.image_data_format() == 'channels_last':
      shape = (16,112,112,3) 
    else:
      shape = (3,16,112,112)
        
    teacher = Sequential()
    teacher.add(Conv3D(64, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    teacher.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    teacher.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    teacher.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    teacher.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    teacher.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    teacher.add(Dropout(0.3))
    teacher.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    teacher.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    teacher.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    teacher.add(Dropout(0.3))
    teacher.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))

    
    teacher.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    teacher.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    teacher.add(Dropout(0.3))
    teacher.add(ZeroPadding3D(padding=(0,1,1)))
    teacher.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))
    
    
    teacher.add(Flatten())
    
    teacher.add(Dense(4096, activation='relu', name='fc6'))
    teacher.add(Dropout(0.5))
    teacher.add(Dense(4096, activation='relu', name='fc7'))
    teacher.add(Dropout(0.5))
    teacher.add(Dense(num_of_classes, activation='softmax', name='fc8'))


    return teacher


def Student_C3D(num_of_classes):
    
    if K.image_data_format() == 'channels_last':
      shape = (16,112,112,3) 
    else:
      shape = (3,16,112,112)
        
    student = Sequential()
    student.add(Conv3D(16, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    student.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    student.add(Conv3D(32, 3, activation='relu', padding='same', name='conv2'))
    student.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    student.add(Conv3D(32, 3, activation='relu', padding='same', name='conv3a'))
    student.add(Conv3D(32, 3, activation='relu', padding='same', name='conv3b'))
    student.add(Dropout(0.2))
    student.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))
   
    
    student.add(Conv3D(64, 3, activation='relu', padding='same', name='conv4a'))
    student.add(Conv3D(64, 3, activation='relu', padding='same', name='conv4b'))
    student.add(Dropout(0.2))
    student.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))
   
    
    student.add(Conv3D(128, 3, activation='relu', padding='same', name='conv5a'))
    student.add(Conv3D(128, 3, activation='relu', padding='same', name='conv5b'))
    student.add(Dropout(0.2))
    student.add(ZeroPadding3D(padding=(0,1,1)))
    student.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))
    
    
    student.add(Flatten())
    
    student.add(Dense(2048, activation='relu', name='fc6'))
    student.add(Dropout(0.5))
    student.add(Dense(1024, activation='relu', name='fc7'))
    student.add(Dropout(0.5))
    student.add(Dense(num_of_classes, activation='softmax', name='fc8'))

    
    return student