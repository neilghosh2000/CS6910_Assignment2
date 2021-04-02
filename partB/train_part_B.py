import math
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
    pre_trained_model = 'InceptionResNetV2',
    batch_size = 32,
    final_dense = 32,
    final_dropout = 0.2,
    k_freeze_percent = 0.6, 
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

data_aug = config.data_aug

if data_aug:
    train_gen = ImageDataGenerator(rotation_range=45, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='reflect',
                                   rescale=1./255, validation_split=0.1)
else:
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


def use_pretrained_model(pre_trained_model, batch_size, final_dense, final_dropout, k_freeze_percent, batch_norm, activation):
    if pre_trained_model == 'InceptionV3':
        model = InceptionV3(input_shape = input_size, include_top = False, weights = 'imagenet') # leaving out the last layer since we have only 10 classes
        k = math.ceil(k_freeze_percent*311)
    elif pre_trained_model == 'InceptionResNetV2':
        model = InceptionResNetV2(input_shape = input_size, include_top = False, weights = 'imagenet')
        k = math.ceil(k_freeze_percent*780)
    elif pre_trained_model == 'ResNet50':
        model = ResNet50(input_shape = input_size, include_top = False, weights = 'imagenet')
        k = math.ceil(k_freeze_percent*175)
    elif pre_trained_model == 'Xception':
        model = Xception(input_shape = input_size, include_top = False, weights = 'imagenet')
        k = math.ceil(k_freeze_percent*132)
        
    
    i = 0
    for layer in model.layers:
        layer.trainable = False
        i+=1
        if(i==k):
            break
        
    x = layers.Flatten()(model.output) # flattening the last layer to a single layer
    
    if activation == "relu":
        x = layers.Dense(final_dense, layers.ReLU())(x)
    elif act_fun == "leaky_relu":
        x = layers.Dense(n_dense, layers.LeakyReLU(alpha=0.1))(x)

    if batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.Dropout(final_dropout)(x) # adding a dropout
    x = layers.Dense(10, tf.nn.softmax)(x) # final softmax function
    
    final_model = Model(model.input,x)
    final_model.summary()
    
    return final_model

pre_trained_model = config.pre_trained_model
batch_size = config.batch_size
final_dense = config.final_dense
final_dropout = config.final_dropout
k_freeze_percent = config.k_freeze_percent
batch_norm = config.batch_norm
activation = config.activation

model = use_pretrained_model(pre_trained_model, batch_size, final_dense, final_dropout, k_freeze_percent, batch_norm, activation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[WandbCallback()])