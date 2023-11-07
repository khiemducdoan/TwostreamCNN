# Imports needed
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 200
img_width = 200
batch_size = 32

# Define your custom CNN block

model = keras.Sequential([
    layers.Input((img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding="same"),
    layers.Conv2D(32, 3, padding="same"),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(29, activation='softmax'),
])

# ImageDataGenerator and flow_from_directory
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    validation_split=0.2,
    dtype= 'float32'
)

train_generator = datagen.flow_from_directory(
    "asl_alphabet_train\\asl_alphabet_train",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",  # Change to "rgb" for color images
    class_mode="sparse",
    shuffle=True,
    subset="training",
    seed=123,
)

validation_generator = datagen.flow_from_directory(
    "asl_alphabet_train\\asl_alphabet_train",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",  # Change to "rgb" for color images
    class_mode="sparse",
    shuffle=True,
    subset="validation",
    seed=123,
)

# Redo model.compile to reset the optimizer states
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# using model.fit (note steps_per_epoch)
model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    verbose=2,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
