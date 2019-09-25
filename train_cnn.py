import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from CNN_mix import read_data

train_data = np.load('Your_train_data')
train_label = np.load('Your _train_label')
test_data = np.load('Your_test_data')
test_label = np.load('Your_test_label')

# transfer to tensor
train_data = tf.reshape(train_data, [-1, 20, 40, 1])
test_data = tf.reshape(test_data, [-1, 20, 40, 1])

kernel_size = 2
pool_size = (2, 2)
strides = (2, 2)
optimizer = 'adam'
loss = 'mean_squared_error'
# batch_size = 10
nb_epoch = 100

model_dir = './model/'
weight_model_filename = 'best_model_weights.hdf5'
architecture_model_filename = 'model_architecture.json'

# The function to optimize
def root_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))**0.5


def make_model(train_data):
    print('Building CNN architecture model..')
    model = Sequential()

    model.add(Convolution2D(20,
                            kernel_size,
                            kernel_size,
                            border_mode='same',
                            input_shape=(20, 40, 1)
                            ))
    model.add(Activation('relu'))
    model.add(Convolution2D(20,
                            kernel_size,
                            kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size,
                           strides=strides))
    model.add(Dropout(0.25))

    model.add(Convolution2D(40,
                            kernel_size,
                            kernel_size,
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(40,
                            kernel_size,
                            kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size,
                           strides=strides))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(320))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[root_mean_squared_error])

    print('Finished building CNN architecture..')
    return model


def train_model(model,
                train_data,
                train_label,
                test_data,
                test_label,
                nb_epoch=100):
    checkpointer = ModelCheckpoint(filepath=model_dir + weight_model_filename,
                                   verbose=1,
                                   save_best_only=True)
    #

    cnn_json_model = model.to_json()
    with open(model_dir + architecture_model_filename, "w") as json_file:
        json_file.write(cnn_json_model)
    print("Saved CNN architecture to disk..")
    print('Start optimizing CNN model..')
    model.fit(train_data,
              train_label,
              batch_size=None,
              steps_per_epoch=500,
              nb_epoch=nb_epoch,
              validation_data=(test_data, test_label),
              validation_steps=137,
              callbacks=[checkpointer],
              shuffle=False,
              verbose=1)


    print('Optimization finished..')
    return model


cnn_model = make_model(train_data)
trained_cnn_model = train_model(cnn_model,
                                train_data,
                                train_label,
                                test_data,
                                test_label,
                                nb_epoch)

trained_cnn_model.save('./best_model_weights.hdf5')

