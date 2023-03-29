import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop

batch_size  = 128
num_classes =  10
epochs      =  20

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
print(train.shape)
print(train.head(10))

X=train.iloc[:,1:].values
y=train.iloc[:,0].values
Z=test.values

X=X/255
Z=Z/255

print(y.shape)

# convert class vectors to binary class matrices
y=keras.utils.to_categorical(y,num_classes=num_classes)
print(y.shape)

model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='RMSprop',
             metrics=['accuracy'])

history=model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)

predictions=model.predict_classes(Z)
print(predictions)
