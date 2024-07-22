import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
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
parser.add_argument('--temperature', type=int, required=True, help="Number of epochs")
parser.add_argument('--source', type=str, required=True, help="pretrained teacher model path")
parser.add_argument('--output_path', type=str, required=True, help="path for saving trained model")
parser.add_argument('--log_path', type=str, required=True, help="path for saving trained history")
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

# Loading finetuned teacher 3DCNN model, pre-trained on sports1M dataset
pre_trained_teacher = keras.models.load_model(args.source) 
pre_trained_teacher.summary()

# Loading student model to be supervised by pre_trained_teacher during knowledge distillation
student_model = networks.Student_C3D(nb_classes)
student_model.summary()

# Initialize and compile distiller
distiller = Distiller(student=student_model, teacher=pre_trained_teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=args.temperature)

distillation_history = distiller.fit(training_generator, epochs=Epochs)
distiller.evaluate(val_generator)

student_model.save(args.output_path)
np.save(args.log_path,distillation_history.history)
