from keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCH = 22
LOSS = 'binary_crossentropy'
Image_Width = 150
Image_Height = 150
Image_Channels = 3
DIMENSIONS = (Image_Width, Image_Height)
DIMENSIONS_WITH_SHAPE = (Image_Width, Image_Width, Image_Channels)
BATCH_SIZE = 250
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

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

# model.add(Conv2D(4, (3, 3), activation='relu', input_shape=DIMENSIONS_WITH_SHAPE))
# model.add(MaxPooling2D(2, 2))

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=DIMENSIONS_WITH_SHAPE))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss=LOSS,
              optimizer=RMSprop(learning_rate=0.002),
              metrics=['accuracy'])

history = model.fit(train_images,
                    epochs=EPOCH,
                    validation_data=val_images,
                    verbose=1)

model.save("model.h5")

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
xc = range(EPOCH)

plt.figure()
plt.plot(xc, train_loss, label="train_loss")
plt.plot(xc, val_loss, label="val_loss")
plt.plot(xc, train_acc, label="train_acc")
plt.plot(xc, val_acc, label="val_acc")
plt.legend()
plt.title("Training loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.savefig("plot.jpg")
plt.show()
