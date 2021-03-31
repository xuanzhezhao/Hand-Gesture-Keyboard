from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization,Embedding,LSTM,Conv1D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D, Flatten, MaxPooling1D,Input,concatenate
from keras.preprocessing import sequence
import numpy as np
import pickle
import os
os.environ['KERAS_BACKEND']='tensorflow'
from matplotlib import pyplot as plt
import keras

pickle_in = open('X_train, X_test,y_train, y_test_sign.pickel', 'rb')
data = pickle.load(pickle_in)
X_train=data[0]
X_test=data[1]
y_train=data[2]
y_test=data[3]

y_train = np_utils.to_categorical(y_train, 26)
y_test = np_utils.to_categorical(y_test, 26)

inputs = Input(shape=(20,))

IMU1 = keras.layers.Lambda(lambda inputs: inputs[:,:5]*1, input_shape=[20])(inputs)
IMU2 = keras.layers.Lambda(lambda inputs: inputs[:,5:10]*1, input_shape=[20])(inputs)
IMU3 = keras.layers.Lambda(lambda inputs: inputs[:,10:15]*1, input_shape=[20])(inputs)
IMU4 = keras.layers.Lambda(lambda inputs: inputs[:,15:20]*1, input_shape=[20])(inputs)

dense1 = Dense(units=32, activation='relu')(IMU1)
dense2 = Dense(units=32, activation='relu')(IMU2)
dense3 = Dense(units=32, activation='relu')(IMU3)
dense4 = Dense(units=32, activation='relu')(IMU4)

merge1=concatenate([dense1,dense2,dense3,dense4])

merge2=Dense(units=128, activation='relu')(merge1)
merge3=Dense(units=64, activation='relu')(merge2)

output=Dense(units=26, activation='softmax')(merge3)
model5 = Model(inputs = inputs, outputs = output)


model5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history5 = model5.fit(X_train, y_train ,batch_size=32, epochs=10, validation_split=0.2)
model5.summary()

score5 = model5.evaluate(X_test, y_test, verbose=1)
print('Validation loss:', score5[0])
print('Validation accuracy:', score5[1])