## To run on CPU instead of GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import  models, optimizers, layers, activations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import wandb
from wandb.keras import WandbCallback

default_parameters = dict(
    data_aug = True,
    batch_size = 64,
    n_filters = 32,
    filter_org = 2,
    dropout = 0.3,
    batch_norm = True,
    activation = "relu"
    )

run = wandb.init(config=default_parameters, project="cs6910_assignment_2", entity="arnesh_neil")
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

test_ds = train_gen.flow_from_directory(
    directory='nature_12K/inaturalist_12K/val/',
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42)

# plt.figure(figsize=(10, 10))
# images, labels = val_ds.next()
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i])
#     plt.title(class_names[np.where(labels[i] == 1)[0][0]])
#     plt.axis("off")
# plt.show()

n_filters = config.n_filters
kernel_size = (3, 3)
act_fun = config.activation
filter_org = config.filter_org
dropout = config.dropout
batch_norm = config.batch_norm

def create_model(n_filters, kernel_size, act_fun, dropout, filter_org, batch_norm):

    model = models.Sequential()
    model.add(layers.Conv2D(n_filters, kernel_size, input_shape=input_size))

    if act_fun == "relu":
        model.add(layers.ReLU())
    elif act_fun == "leaky_relu":
        model.add(layers.LeakyReLU(alpha=0.1))

    if batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    for i in range(4):
        model.add(layers.Dropout(dropout))
        model.add(layers.Conv2D((filter_org**(i+1))*n_filters, kernel_size))
        if act_fun == "relu":
            model.add(layers.ReLU())
        elif act_fun == "leaky_relu":
            model.add(layers.LeakyReLU(alpha=0.1))

        if batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    n_dense = 32

    model.add(layers.Flatten())

    if act_fun == "relu":
        model.add(layers.Dense(n_dense, layers.ReLU()))
    elif act_fun == "leaky_relu":
        model.add(layers.Dense(n_dense, layers.LeakyReLU(alpha=0.1)))

    if batch_norm:
        model.add(layers.BatchNormalization())

    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, tf.nn.softmax))

    model.summary()

    return model


model = create_model(n_filters, kernel_size, act_fun, dropout, filter_org, batch_norm)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[WandbCallback()])

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')

