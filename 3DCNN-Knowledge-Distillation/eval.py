# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:02:28 2023

@author: hayatu
"""

import tensorflow as tf
from tensorflow import keras
from keras import Model
import numpy as np
import networks
from data_generator import TrainGenerator, ValGenerator, PredictGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import scikitplot as skplt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, auc
import matplotlib.pyplot as plt
from itertools import cycle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help="path to frames dataset")
parser.add_argument('--annotations', type=str, required=True, help="path to labels")
parser.add_argument('--recognizer', type=str, required=True, help="path to the trained model")
args = parser.parse_args()

dataset_frames = np.load(args.data)
dataset_labels = np.load(args.annotations)

# Split frames and labels into training and testing data with 80% training and 20% testing data
_, test_x, _, test_y = train_test_split(dataset_frames, dataset_labels, test_size = 0.20, shuffle = False)


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model = keras.models.load_model(args.recognizer,compile=False)
model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = opt, metrics = ["accuracy"])

eva_score = model.evaluate(test_x, test_y)


