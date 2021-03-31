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

X_train=np.reshape(X_train,[3741,360,1])
X_test=np.reshape(X_test,[1604,360,1])

# model3 is for lstm
lstm_model = Sequential()
lstm_model.add(LSTM(256,return_sequences=True))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(27,activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history3 = lstm_model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.2)
lstm_model.summary()
print('Final test accuracy of LSTM is:')
score3 = lstm_model.evaluate(X_test, y_test)
print('lstm:',score3[1])