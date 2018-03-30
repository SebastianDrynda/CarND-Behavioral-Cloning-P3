import cv2
import csv
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


def get_driving_log(directory, skip_header=True):
    """
    Returns the driving log data lines from the directory `directory`.
    If the file include a header line, pass `skipHeader=True`.
    """
    lines = []
    with open(directory + '/driving_log.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        if skip_header:
            next(csv_reader, None)
        for line in csv_reader:
            lines.append(line)
            
    return lines


def get_data(data_paths, test_size=0):
    """
    Returns all the images and steering data on the paths `data_paths`.
    Returns `([center_img_paths], [left_img_paths], [right_img_paths], [measurements])`
    """
    center_img_paths = []
    left_img_paths = []
    right_img_paths = []
    measurements = []
    
    for directory in data_paths:
        lines = get_driving_log(directory)  
        print('directory ', directory)
        i = 0;
        for line in lines:
            center_img_paths.append(directory + '/IMG/' + line[0].strip().split('/')[-1])
            left_img_paths.append(directory + '/IMG/' + line[1].strip().split('/')[-1])
            right_img_paths.append(directory + '/IMG/' + line[2].strip().split('/')[-1])
            measurements.append(np.float64(line[3]))
            
            # For testing small data sets on local machine 
            if test_size > 0:
                i += 1
                if i == test_size:
                    break

    return (center_img_paths, left_img_paths, right_img_paths, measurements)


def combine_images(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([image_paths], [measurements])
    """
    image_paths = []
    image_paths.extend(center)
    image_paths.extend(left)
    image_paths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    
    return (image_paths, measurements)


def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`img_paths`, `measurement`).
    
    Executing the following steps:
      - Converts the image from BGR to RGB.
      - Adds the image and `measurement` to `images` and `measurements`.
      - Flips the image vertically.
      - Inverts the sign of the `measurement`.
      - Adds the flipped image and inverted `measurement` to `images` and `measurements`.
    """
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for img_paths, measurement in batch_samples:
                original_img = cv2.imread(img_paths)
                image = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                images.append(image)
                measurements.append(measurement)
                # Flipping
                images.append(cv2.flip(image, 1))
                measurements.append(measurement * -1.0)

            # trim image to only see section with road
            X_input = np.array(images)
            y_output = np.array(measurements)
            yield sklearn.utils.shuffle(X_input, y_output)


def preprocessing_layers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model


def lenet_model():
    """
    Creates a LeNet model.
    """
    model = preprocessing_layers()
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


def nvidia_model():
    """
    Creates NVidia Autonomous Car model
    """
    model = preprocessing_layers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# Reading images locations
data_paths = ['data/01_udacity', 'data/02_track1_forward', 'data/03_track1_backward', 'data/04_track2_forward', 'data/05_track2_forward']
center_img_paths, left_img_paths, right_img_paths, measurements = get_data(data_paths, test_size=0)

# Combine images and correct steering data by factor 0.2
correction_factor = 0.2
img_paths, measurements = combine_images(center_img_paths, left_img_paths, right_img_paths, measurements, correction_factor)

print('Total Images: {}'.format(len(img_paths)))

# Splitting samples (80% for training samples and 20% for validation samples)
samples = list(zip(img_paths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Training samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

# Creating generators with batch size 32
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#  LeNet model
# print('Using LeNet model for training')
# model = lenet_model()
# model_name = 'model_lenet'

#  NVidia Autonomous Car model
print('Using NVidia Autonomous Car model for training')
model = nvidia_model()
model_name = 'model_nvidia'

# Compiling and training the model
epochs = 2
verbose = 1
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=epochs, verbose=verbose)

model.save('models/' + model_name + '_epochs_' + str(epochs) + '.h5')
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

from keras.utils import plot_model
plot_model(model, to_file='model.png')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('images/model_mse_lost.png') 
# plt.show()

model.summary()
