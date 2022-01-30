import tensorflow as tf
from keras.layers import Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Conv2D
import matplotlib.pyplot as plt

EPOCH = 100
LOSS = 'binary_crossentropy'
OPTIMIZER = 'adam'
Image_Width = 150
Image_Height = 150
DIMENSIONS = (Image_Width, Image_Height)
Image_Channels = 3
BATCH_SIZE = 50
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
categories = ['Cat', 'Dog']

train_data_generator = ImageDataGenerator(rotation_range=20,
                                          rescale=1. / 255,
                                          shear_range=0.1,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1)

validation_data_generator = ImageDataGenerator(rotation_range=20,
                                               rescale=1. / 255,
                                               shear_range=0.1,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1)

train_images = train_data_generator.flow_from_directory(os.path.join(DATASET_PATH, "train"),
                                                        target_size=DIMENSIONS,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary')

val_images = validation_data_generator.flow_from_directory(os.path.join(DATASET_PATH, "validation"),
                                                           target_size=DIMENSIONS,
                                                           batch_size=BATCH_SIZE,
                                                           class_mode='binary')

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=LOSS,
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit_generator(
    train_images,
    # steps_per_epoch=40,
    epochs=EPOCH,
    validation_data=val_images,
    # validation_steps=10,
    verbose=2
)

model.save("model.h5")
