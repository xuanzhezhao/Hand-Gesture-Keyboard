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

# model2 is for CNN
model2 = Sequential()
model2.add(Conv1D(32, 3, padding="same"))
model2.add(Activation('relu'))
model2.add(MaxPooling1D(2,2))

model2.add(Flatten())
model2.add(Dense(27,activation='softmax'))
model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history2 = model2.fit(X_train, y_train, batch_size=32, epochs=5,  validation_split=0.2)
model2.summary()
print('Final test accuracy of CNN is:')
score2= model2.evaluate(X_test, y_test)
print("CNN:",score2[1])