from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization,Embedding,LSTM,Conv1D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D, Flatten, MaxPooling1D
from keras.preprocessing import sequence
import numpy as np
import pickle
from matplotlib import pyplot as plt

pickle_in = open('X_train, X_test,y_train, y_test.pickel', 'rb')
data = pickle.load(pickle_in)
X_train=data[0]
X_test=data[1]
y_train=data[2]
y_test=data[3]

y_train = np_utils.to_categorical(y_train, 27)
y_test = np_utils.to_categorical(y_test, 27)

# model1 is for MLP
model1 = Sequential()
model1.add(Dense(128,activation='relu'))
model1.add(Dense(64,activation='relu'))
model1.add(Dense(27,activation='softmax'))
model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history1 = model1.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.2)
model1.summary()
print('Final test accuracy of MLP is:')
score1 = model1.evaluate(X_test, y_test)
print("MLP:",score1[1])