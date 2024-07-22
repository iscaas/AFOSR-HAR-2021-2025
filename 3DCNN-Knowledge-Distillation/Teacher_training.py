# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:48:15 2022

@author: hayatu
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
#import keras.backend as K
from data_generator import TrainGenerator, ValGenerator, PredictGenerator
from keras import Model
import math
import networks
import os
import datetime
from distiller import Distiller
from keras.layers.core import Dense, Dropout, Flatten
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="path to frames dataset")
parser.add_argument('--annotations', type=str, required=True, help="path to labels")
parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
parser.add_argument('--output_path', type=str, required=True, help="path for saving trained model")
parser.add_argument('--log_path', type=str, required=True, help="path for saving trained history")
parser.add_argument('--task', type=str, required=True, help="finetuning pre-trained C3D -> use [fintuning_C3D_Sports1M] or train C3D locally -> use [train_local_teacher]")
args = parser.parse_args()


dataset_frames = np.load(args.data)
dataset_labels = np.load(args.annotations)

# Split frames and labels into training and testing data with 80% training and 20% testing data
train_x, test_x, train_y, test_y = train_test_split(dataset_frames, dataset_labels, test_size = 0.20, shuffle = True)

batch_size = args.batch_size
Epochs = args.epochs
training_generator = TrainGenerator(train_x, train_y, batch_size)
val_generator = TrainGenerator(test_x, test_y, batch_size)

nb_classes = len(np.unique(dataset_labels, return_counts=True)[0])

if args.task == "fintuning_C3D_Sports1M":
  teacher = networks.C3D(weights='sports1M')

  # freeze the first 19 layers 
  for layer in teacher.layers[:19]:
    layer.trainable = False

  for i, layer in enumerate(teacher.layers):
    print(i, layer.name, "-", layer.trainable)

  print(teacher.layers[-2].output)

  X = teacher.layers[-2].output
  predictions = Dense(nb_classes, activation='softmax')(X)
  teacher = Model(teacher.input, predictions)
  teacher.summary()
elif args.task == "train_local_teacher":
  teacher = networks.Teacher_C3D(nb_classes)
else:
  print("Task is not Specified.....")

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Compile the model and specify loss function, optimizer and metrics values to the model
teacher.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = opt, metrics = ["accuracy"])


model_training_history = teacher.fit(
    training_generator,
    steps_per_epoch=math.floor(len(train_x) // batch_size),
    validation_data=val_generator,
    validation_steps=math.floor(len(test_x) // batch_size),
    verbose=1,
    epochs=Epochs,
)
test_history = teacher.evaluate(val_generator)

teacher.save(args.output_path)
np.save(args.log_path,model_training_history.history)




