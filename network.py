import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.regularizers import l2
from keras.initializers import normal
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')

def schedule(epoch,old_rate):
    if epoch%6 == 0:
        return old_rate/10
    else:
        return old_rate

datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True) #GCN
datagen.fit(x_train)

model = Sequential()

model.add(Conv2D(64,kernel_size=4,activation='relu',input_shape=(27,27,3)))
model.add(Conv2D(64,kernel_size=3,activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128,kernel_size=3,activation='relu',padding='same'))
model.add(Conv2D(128,kernel_size=3,activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_initializer=normal(0,0.01),kernel_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu',kernel_initializer=normal(0,0.01),kernel_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Dense(2,activation='sigmoid',kernel_initializer=normal(0,0.01)))

model.compile(SGD(lr=0.001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train,y_train,batch_size=256),steps_per_epoch=int(x_train.shape[0]/256),epochs=19,callbacks=[LearningRateScheduler(schedule,verbose=1),ModelCheckpoint('gcn_checkpoint.keras')])
model.save('gcn.keras')
