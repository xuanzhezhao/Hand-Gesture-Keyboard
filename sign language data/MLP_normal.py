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

model6 = Sequential()
model6.add(Dense(128,activation='relu'))
model6.add(Dense(128,activation='relu'))
model6.add(Dense(64,activation='relu'))
model6.add(Dense(26,activation='softmax'))
model6.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history6 = model6.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
print('Final test accuracy MLP is:')
score6 = model6.evaluate(X_test, y_test)
print("MLP:",score6[1])