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

#reading test and train and dev files
#///////////////
#def read(file_path):
#     file = open(file_path,"rb")
#     shape = np.fromfile(file,dtype=np.int16,count=2)
#     data = np.fromfile(file,dtype=np.float32,count=-1)
#     data = data.reshape(( shape[0], 128,64))
#     file.close()
#     return data
#trainfile = open('filelist.train.rnd-select.txt', "r")
#testfile=open('filelist.test.rnd-select.txt', "r")
#devfile=open('filelist.dev.rnd-select.txt', "r")
#count=0
#for line in trainfile:
#     y = line.split()
#     for i in y:
#         count += 1
#         print(count)
#         x = str(i)
#         print(x[10:], 'Alexa_64x128\\' + x[10:-4] + 'bin', 'Alexa\\' + x[10:])
#         if count is 1 :
#             inp=read('Alexa_64x128\\' + x[10:-4] + 'bin')
#             #target
#             with hp.File('Alexa\\' + x[10:]) as h5:
#                 dataset=h5.get('spectral_parameters')
#                 target=np.array(dataset)
#         else:
#             data = read('Alexa_64x128\\' + x[10:-4] + 'bin')
#             inp = np.concatenate((inp, data), axis=0)
#             # target
#             with hp.File('Alexa\\' + x[10:]) as h5:
#                 dataset = h5.get('spectral_parameters')
#                 data2 = np.array(dataset)
#                 target=np.concatenate((target,data2),axis=0)
#print(inp.shape)
#print(target.shape)
#print(count, 'train files read')
#count=0
# #read testfile
#for line in devfile:
#     y = line.split()
#     for i in y:
#         count += 1
#         print(count)
#         x = str(i)
#         print(x[10:])
#         if count is 1 :
#             devinp=read('Alexa_64x128\\' + x[10:-4] + 'bin')
#             #target
#             with hp.File('Alexa\\' + x[10:]) as h5:
#                 dataset=h5.get('spectral_parameters')
#                 devtarget=np.array(dataset)
#         else:
#             data = read('Alexa_64x128\\' + x[10:-4] + 'bin')
#             devinp = np.concatenate((devinp, data), axis=0)
#             # target
#             with hp.File('Alexa\\' + x[10:]) as h5:
#                 dataset = h5.get('spectral_parameters')
#                 data2 = np.array(dataset)
#                 devtarget=np.concatenate((devtarget,data2),axis=0)
#print(devinp.shape)
#print(devtarget.shape)
#print(count, 'dev files read')
#count=0
#for line in testfile:
#     y = line.split()
#     for i in y:
#         count += 1
#         print(count)
#         x = str(i)
#         print(x[10:])
#         if count is 1 :
#             tstinp=read('Alexa_64x128\\' + x[10:-4] + 'bin')
#             #target
#             with hp.File('Alexa\\' + x[10:]) as h5:
#                 dataset=h5.get('spectral_parameters')
#                 tsttarget=np.array(dataset)
#         else:
#             data = read('Alexa_64x128\\' + x[10:-4] + 'bin')
#             tstinp = np.concatenate((tstinp, data), axis=0)
#             # target
#             with hp.File('Alexa\\' + x[10:]) as h5:
#                 dataset = h5.get('spectral_parameters')
#                 data2 = np.array(dataset)
#                 tsttarget=np.concatenate((tsttarget,data2),axis=0)
#print(tstinp.shape)
#print(tsttarget.shape)
#print(count, 'test files read')
# # ///////////
#
#
#
# writing to files
#with hp.File('TrainInputFile.h5','w') as h5:
#     h5.create_dataset('Xvalues',data=inp)


#with hp.File('TrainTargetFile.h5','w') as h5:
#     h5.create_dataset('Yvalues',data=target)


#with hp.File('DevInputFile.h5','w') as h5:
#     h5.create_dataset('Xvalues',data=devinp)

#with hp.File('DevTargetFile.h5','w') as h5:
#     h5.create_dataset('Yvalues',data=devtarget)

#with hp.File('TestInputFile.h5','w') as h5:
#     h5.create_dataset('Xvalues',data=tstinp)

#with hp.File('TestTargetFile.h5','w') as h5:
#     h5.create_dataset('Yvalues',data=tsttarget)

#readfile
with hp.File('E:/university/UniSzegedProjects/DataSets/dataset/TrainInputFile.h5','r') as h5:
    dataset=h5.get('Xvalues')
    X_train=np.array(dataset)

with hp.File('E:/university/UniSzegedProjects/DataSets/dataset/out-waveGlow/TrainTargetsWaveglow.h5','r') as h5:
#with hp.File('TrainTargetFile.h5','r') as h5:
    dataset=h5.get('Yvalues')
    y_train=np.array(dataset)



with hp.File('E:/university/UniSzegedProjects/DataSets/dataset/DevInputFile.h5','r') as h5:
    dataset=h5.get('Xvalues')
    X_dev=np.array(dataset)

with hp.File('E:/university/UniSzegedProjects/DataSets/dataset/out-waveGlow/DevTargetsWaveglow.h5','r') as h5:
#with hp.File('DevTargetFile.h5','r') as h5:
    dataset=h5.get('Yvalues')
    y_dev=np.array(dataset)

with hp.File('E:/university/UniSzegedProjects/DataSets/dataset/TestInputFile.h5','r') as h5:
    dataset=h5.get('Xvalues')
    X_test=np.array(dataset)


with hp.File('E:/university/UniSzegedProjects/DataSets/dataset/out-waveGlow/TestTargetsWaveglow.h5','r') as h5:
#with hp.File('TestTargetFile.h5','r') as h5:
    dataset=h5.get('Yvalues')
    y_test=np.array(dataset)

#x_scaler = NDStandardScaler()
#X_train = x_scaler.fit_transform(X_train)
#X_dev= x_scaler.transform(X_dev)
#X_test= x_scaler.transform(X_test)
X_train -= 0.5
X_train *= 2
X_dev -= 0.5
X_dev *= 2
X_test -= 0.5
X_test *= 2

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_dev= y_scaler.transform(y_dev)
y_test= y_scaler.transform(y_test)
#print(y_scaler.mean_)

#this is required for convolutional networks
shape = X_train.shape + (1,)     
X_train = X_train.reshape(shape) 
shape = X_dev.shape + (1,)     
X_dev = X_dev.reshape(shape) 
shape = X_test.shape + (1,)     
X_test = X_test.reshape(shape) 

print(X_train.shape)
model=Sequential()
model.add(InputLayer(input_shape=X_train.shape[1:]))
#model.add(Conv2D(filters=50, kernel_size=(7,7), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation="swish", use_bias=True, kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None), kernel_constraint=max_norm(0.5), bias_constraint=None))
#model.add(MaxPooling2D(pool_size=(4,4))) 
#model.add(BatchNormalization()) 
#model.add(Conv2D(filters=50, kernel_size=(7,7), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation="swish", use_bias=True, kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None), kernel_constraint=max_norm(0.5), bias_constraint=None))
#model.add(MaxPooling2D(pool_size=(2,2))) 
#model.add(BatchNormalization()) 
#model.add(Conv2D(filters=30, kernel_size=(7,7), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation="swish", use_bias=True, kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None), kernel_constraint=max_norm(0.5), bias_constraint=None))
#model.add(MaxPooling2D(pool_size=(2,2))) 
#model.add(Dropout(0.2)) 
#model.add(BatchNormalization()) 
#model.add(Flatten())
#model.add(Dense(1000,activation='swish', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None), kernel_constraint=max_norm(0.75)))#,kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.0001)))
#model.add(Dropout(0.2)) 
#model.add(BatchNormalization()) 
#model.add(Dense(1000,activation='swish', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None), kernel_constraint=max_norm(0.75)))#,kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.0001)))
#model.add(Dropout(0.2)) 
#model.add(BatchNormalization()) 
#adam-fele
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

history=model.fit(X_train,y_train,batch_size=100,epochs=40,verbose=1,callbacks =[earlystopper, lrr],shuffle=False,validation_data=(X_dev,y_dev))
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
y_predict_train=model.predict(X_train)
y_predict_test=model.predict(X_test)
y_predict_dev=model.predict(X_dev)
#print(y_predict_test[0,:])
#print(y_predict_dev[0,:])
#print(y_test[0,:])
#print(y_dev[0,:])
print('r2 score on train set is :\t{:0.3f}'.format(r2_score(y_train,y_predict_train)))
print('r2 score on dev data is:\t{:0.3f}'.format(r2_score(y_dev,y_predict_dev)))
print('r2 score on test data is:\t{:0.3f}'.format(r2_score(y_test,y_predict_test)))

print('train-mean squared error:\t{:0.3f}'.format(mean_squared_error(y_train,y_predict_train)))
print('tdev-mean squared error:\t{:0.3f}'.format(mean_squared_error(y_dev, y_predict_dev)))
print('test-mean squared error:\t{:0.3f}'.format(mean_squared_error(y_test, y_predict_test)))
