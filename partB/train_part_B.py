import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  models, optimizers, layers, activations
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet50, Xception

import wandb
from wandb.keras import WandbCallback

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

default_parameters = dict(
    data_aug = True,
    pre_trained_model = 'InceptionV3',
    batch_size = 32,
    final_dense = 32,
    final_dropout = 0.2,
    batch_norm = True,
    activation = "relu"
    )

run = wandb.init(config= default_parameters,project="cs6910_Assignment2", entity="arneshbose1")
config = wandb.config

image_size = (128,128)
input_size = (128,128,3)
batch_size = config.batch_size

class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi',
               'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']

train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_ds = train_gen.flow_from_directory(
    directory='nature_12K/inaturalist_12K/train/',
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset="training")

val_ds = train_gen.flow_from_directory(
    directory='nature_12K/inaturalist_12K/train/',
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset="validation")

test_ds = train_gen.flow_from_directory(
    directory='nature_12K/inaturalist_12K/val/',
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42)

def use_pretrained_model(pre_trained_model, batch_size, final_dense, final_dropout, batch_norm, activation):
    if pre_trained_model == 'InceptionV3':
        model = InceptionV3(input_shape = input_size, include_top = False, weights = 'imagenet') # leaving out the last layer since we have only 10 classes
    
    for layer in model.layers:
        layer.trainable = False
        
    x = layers.Flatten()(model.output) # flattening the last layer to a single layer
    x = layers.Dense(final_dense, activation = activation)(x) # adding a dense layer at the end
    x = layers.Dropout(final_dropout)(x) # adding a dropout
    x = layers.Dense(10, tf.nn.softmax)(x) # final softmax function
    
    final_model = Model(model.input,x)
    final_model.summary()
    
    return final_model

pre_trained_model = config.pre_trained_model
batch_size = config.batch_size
final_dense = config.final_dense
final_dropout = config.final_dropout
batch_norm = config.batch_norm
activation = config.activation

model = use_pretrained_model(pre_trained_model, batch_size, final_dense, final_dropout, batch_norm, activation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[WandbCallback()])