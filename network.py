from patch import Patch
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.regularizers import l2
from keras.initializers import normal
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

def schedule(epoch,old_rate):
    if epoch%6 == 0:
        return old_rate/10
    else:
        return old_rate

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
model.fit(x_train,y_train,batch_size=256,epochs=19,callbacks=[LearningRateScheduler(schedule,verbose=1),ModelCheckpoint('plain_checkpoint.keras')])
model.save('plain.keras')
