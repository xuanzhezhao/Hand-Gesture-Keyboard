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

X_train=np.reshape(X_train,[3141,20,1])
X_test=np.reshape(X_test,[786,20,1])

model4 = Sequential()
model4.add(Conv1D(16, 3,padding="same"))
model4.add(Activation('relu'))

model4.add(Flatten())
model4.add(Dense(26,activation='softmax'))
model4.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history4 = model4.fit(X_train, y_train, batch_size=36, epochs=10, validation_split=0.2)
print('Final test accuracy of CNN is:')
score4= model4.evaluate(X_test, y_test)
print('CNN:',score4[1])