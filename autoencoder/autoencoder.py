import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import data_process

#Load sign language dataset
data, label, a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z= data_process.load_data()
data=data_process.normalize_data(data)
#data processing


x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.3)
num_classes = 26
#y_train=to_categorical(y_train,num_classes)
#y_test=to_categorical(y_test,num_classes)


#print(x_train.shape)
#print(y_train.shape)
print(type(y_test))
#print(y_test)
#Autoencoder
model = Sequential()
model.add(Dense(16,input_shape=(20,)))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(3, name='representation'))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])
print(model.summary())
epochs = 200
validation_split = 0.2
history = model.fit(data, data, batch_size=128,
          epochs=epochs, validation_split=validation_split)

def predict_representation(model, data, layer_name='representation'):
  ## We form a new model. Instead of doing \psi\phi(x), we only take \phi(x)
  ## To do so, we use the layer name
  intermediate_layer_model = Model(inputs=model.input,
                                   outputs=model.get_layer(layer_name).output)
  representation = intermediate_layer_model.predict(data)
  representation = representation.reshape(representation.shape[0], -1)
  return representation

representation = predict_representation(model, data)

# np.savetxt("a.txt", representation[0:a])
# np.savetxt("b.txt", representation[a:a+b])
# np.savetxt("c.txt", representation[a+b:a+b+c])
# np.savetxt("d.txt", representation[a+b+c:a+b+c+d])
# np.savetxt("e.txt", representation[a+b+c+d:a+b+c+d+e])
# np.savetxt("f.txt", representation[a+b+c+d+e:a+b+c+d+e+f])
# np.savetxt("g.txt", representation[a+b+c+d+e+f:a+b+c+d+e+f+g])
# np.savetxt("h.txt", representation[a+b+c+d+e+f+g:a+b+c+d+e+f+g+h])
# np.savetxt("i.txt", representation[a+b+c+d+e+f+g+h:a+b+c+d+e+f+g+h+i])
# np.savetxt("j.txt", representation[a+b+c+d+e+f+g+h+i:a+b+c+d+e+f+g+h+i+j])
# np.savetxt("k.txt", representation[a+b+c+d+e+f+g+h+i+j:a+b+c+d+e+f+g+h+i+j+k])
# np.savetxt("l.txt", representation[a+b+c+d+e+f+g+h+i+j+k:a+b+c+d+e+f+g+h+i+j+k+l])
# np.savetxt("m.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l:a+b+c+d+e+f+g+h+i+j+k+l+m])
# np.savetxt("n.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m:a+b+c+d+e+f+g+h+i+j+k+l+m+n])
# np.savetxt("o.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o])
# np.savetxt("p.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p])
# np.savetxt("q.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q])
# np.savetxt("r.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r])
# np.savetxt("s.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s])
# np.savetxt("t.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t])
# np.savetxt("u.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u])
# np.savetxt("v.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v])
# np.savetxt("w.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w])
# np.savetxt("x.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x])
# np.savetxt("y.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y])
# np.savetxt("z.txt", representation[a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y:a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y+z])
def plot_representation_label(representation, labels, plot3d=1):
    ## Function used to plot the representation vectors and assign different
    ## colors to the different classes

    # First create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    # In case representation dimension is 3, we can plot in a 3d projection too
    if plot3d:
        ax = fig.add_subplot(111, projection='3d')

    # Check number of labels to separate by colors
    #n_labels = labels.max() + 1
    #n_labels=n_labels.astype('int32')
    n_labels=26
    # Color map, and give different colors to every label
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / (n_labels)) for i in range(n_labels)])
    # Loop is to plot different color for each label
    for l in range(n_labels):
        # Only select indices for corresponding label
        index = labels == l
        ind=index.reshape((len(index,)))
        print(ind.shape)
        if plot3d:
            ax.scatter(representation[ind, 0], representation[ind, 1],
                       representation[ind, 2], label=str(l), s=20)
        else:
            ax.scatter(representation[ind, 0], representation[ind, 1], label=str(l))
    ax.legend()
    plt.title('Features in the representation space with corresponding label')
    plt.show()
    return fig, ax


plot_representation_label(representation, label)

