import pandas as pds
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,InputLayer,Dropout,Conv2D,MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import adam,adadelta,SGD
from keras import regularizers 
from keras.constraints import unit_norm, max_norm 
from sklearn import preprocessing
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

import h5py as hp

import numpy as np
#import cv2

from keras.backend import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


#readfile
with hp.File('./dataset/TrainInputFile.h5','r') as h5:
    dataset=h5.get('Xvalues')
    train_input=np.array(dataset)

with hp.File('./dataset/TrainTargetsWaveglow.h5','r') as h5:
#with hp.File('TrainTargetFile.h5','r') as h5:
    dataset=h5.get('Yvalues')
    train_output=np.array(dataset)



with hp.File('./dataset/DevInputFile.h5','r') as h5:
    dataset=h5.get('Xvalues')
    val_input=np.array(dataset)

with hp.File('./dataset/DevTargetsWaveglow.h5','r') as h5:
#with hp.File('DevTargetFile.h5','r') as h5:
    dataset=h5.get('Yvalues')
    val_output=np.array(dataset)

with hp.File('./dataset/TestInputFile.h5','r') as h5:
    dataset=h5.get('Xvalues')
    test_input=np.array(dataset)


with hp.File('./dataset/TestTargetsWaveglow.h5','r') as h5:
#with hp.File('TestTargetFile.h5','r') as h5:
    dataset=h5.get('Yvalues')
    test_output=np.array(dataset)

# standardization
train_input -= 0.5
train_input *= 2
val_input -= 0.5
val_input *= 2
test_input -= 0.5
test_input *= 2

y_scaler = StandardScaler()
train_output = y_scaler.fit_transform(train_output)
val_output= y_scaler.transform(val_output)
test_output= y_scaler.transform(test_output)
#print(y_scaler.mean_)

#adding a dimention for 2d model
shape = train_input.shape + (1,)
train_input = train_input.reshape(shape)
shape = val_input.shape + (1,)
val_input = val_input.reshape(shape)
shape = test_input.shape + (1,)
test_input = test_input.reshape(shape)

# print(test_input.shape)
model=Sequential()
model.add(InputLayer(input_shape=train_input.shape[1:]))

model.add(Conv2D(filters=30,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
model.add(Dropout(0.2)) 
model.add(Conv2D(filters=60,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None) ,kernel_regularizer=regularizers.l1(0.00001)))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=90,kernel_size=(13,13),strides=(2,1),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
model.add(Dropout(0.2)) 
model.add(Conv2D(filters=120,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500,activation='swish', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None),kernel_regularizer=regularizers.l1(0.000005)))#, kernel_regularizer=regularizers.l1(0.00001)))#,kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.0001)))
model.add(Dropout(0.2)) 
model.add(Dense(80,activation='linear'))#, kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None)))
#SGD(lr=0.1),adam(lr=0.001,  epsilon=1e-08, decay=0.0)
model.compile(SGD(lr=0.1,  momentum=0.1, nesterov=True),loss='mean_squared_error', metrics=['mean_squared_error'])
earlystopper = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=2, verbose=1, mode='auto')
lrr = ReduceLROnPlateau(monitor='val_mean_squared_error', patience=1, verbose=1, factor=0.5, min_lr=0.0001) 
#
#
model.summary()

history=model.fit(train_input,train_output,batch_size=100,epochs=40,verbose=1,callbacks =[earlystopper, lrr],shuffle=False,validation_data=(val_input,val_output))
#
#
history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
#
#plt.plot(loss_values,'bo',label='training loss')
#plt.plot(val_loss_values,'r',label='training loss val')
#plt.show()
#
y_predict_train=model.predict(train_input)
y_predict_test=model.predict(test_input)
y_predict_dev=model.predict(val_input)

print('r2 score on train set is :\t{:0.3f}'.format(r2_score(train_output,y_predict_train)))
print('r2 score on dev data is:\t{:0.3f}'.format(r2_score(val_output,y_predict_dev)))
print('r2 score on test data is:\t{:0.3f}'.format(r2_score(test_output,y_predict_test)))

print('train-mean squared error:\t{:0.3f}'.format(mean_squared_error(train_output,y_predict_train)))
print('tdev-mean squared error:\t{:0.3f}'.format(mean_squared_error(val_output, y_predict_dev)))
print('test-mean squared error:\t{:0.3f}'.format(mean_squared_error(test_output, y_predict_test)))
