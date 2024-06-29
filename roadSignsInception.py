# Import libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications import InceptionV3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Transfer Learning
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import Xception



# We're using keras' ImageDataGenerator class to load our image data.
# See (https://keras.io/api/preprocessing/image/#imagedatagenerator-class) for details
#
# A couple of things to note:
# 1. We're specifying a number for the seed, so we'll always get the same shuffle and split of our images.
# 2. Class names are inferred automatically from the image subdirectory names.
# 3. We're splitting the training data into 80% training, 20% validation.


training_dir = '/home/cleb/training/'
image_size = (100, 100)

# Split up the training data images into training and validations sets
# We'll use and ImageDataGenerator to do the splits
# ImageDataGenerator can also be used to do preprocessing and agumentation on the files as can be seen with rescale

train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=.2,
        # rotation_range=20 # adds rotation augmentation to training images
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        zoom_range=0.2,
        # brightness_range=[0.8, 1.2]
        )
validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=.2
        )

train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size = image_size,
        subset="training",
        batch_size=32,
        class_mode='sparse',
        seed=42,shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
        training_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='sparse',
        subset="validation",
        seed=42)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


epochs = 60
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
#     ModelCheckpoint("Bst_model.h5", save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# save the model
# model.save('inception_model_brightness.h5')

test_dir = '/home/cleb'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        classes=['mini_holdout'],
        target_size=image_size,
        class_mode='sparse',
        shuffle=False)
probabilities = model.predict(test_generator)
predicted_labels = [np.argmax(probas) for probas in probabilities]

# Load the CSV file
csv_file_path = '/home/cleb/mini_holdout_answers.csv'
df = pd.read_csv(csv_file_path)

# Extract the actual labels
true_labels = df['ClassId'].values

# Assuming you have your predicted labels in an array
# Replace this with your actual predicted labels array
# predicted_labels = np.array([...])

# Ensure that the length of predicted labels matches the length of true labels
assert len(predicted_labels) == len(true_labels), "Mismatch in number of predictions and actual labels"

# Calculate the number of correct predictions
correct_predictions = (predicted_labels == true_labels).sum()
total_predictions = len(true_labels)

# Calculate accuracy
accuracy = correct_predictions / total_predictions



print(f'\nPredictions: {predicted_labels}')
print(f'\nAccuracy:    {accuracy}')