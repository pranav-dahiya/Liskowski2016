import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.regularizers import l2
from keras.initializers import normal
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os


def schedule(epoch, old_rate):
    if epoch == 6 or epoch == 12 or epoch == 18:
        return old_rate/10
    else:
        return old_rate


def train_model(name, x_train, y_train):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(27, 27, 3)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    if name != 'nopool':
        model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    if name != 'nopool':
        model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer=normal(0, 0.01), kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer=normal(0, 0.01), kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid', kernel_initializer=normal(0, 0.01)))

    model.compile(SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    #model = load_model('nopool_checkpoint.keras')
    model.fit(x_train, y_train, batch_size=256, epochs=19, callbacks=[LearningRateScheduler(schedule, verbose=1), ModelCheckpoint('data/'+model_name+'_checkpoint.keras')])
    model.save('data/'+model_name+'.keras')

    os.remove('data/'+model_name+'_checkpoint.keras')


x_train = np.memmap('data/x_train.npy', dtype=np.uint8, mode='r', shape=(400000, 49, 49, 3))
y_train = np.memmap('data/y_train.npy', dtype=np.uint8, mode='r', shape=(400000, 2))
x_train = x_train[:,11:38,11:38,:]
train_model('plain', x_train, y_train)

train_model('nopool', x_train, y_train)

del x_train
x_train = np.memmap('data/x_gcn_train.npy', dtype=np.float32, mode='r', shape=(400000, 27, 27, 3))
train_model('gcn', x_train, y_train)

del x_train
x_train = np.memmap('data/x_zca_train.npy', dtype=np.float32, mode='r', shape=(400000, 27, 27, 3))
train_model('zca', x_train, y_train)

del x_train
del y_train
x_train = np.memmap('data/x_balanced_train.npy', dtype=np.uint8, mode='r', shape=(400000, 27, 27, 3))
y_train = np.memmap('data/y_balanced_train.npy', dtype=np.uint8, mode='r', shape=(400000, 2))
train_model('balanced', x_train, y_train)
