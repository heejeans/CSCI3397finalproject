#Adapted from syamkakarla98

import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import matplotlib.image as image
from tqdm import tqdm
from PIL import Image

from datetime import datetime 

import cv2
import numpy as np
import os
from tqdm import tqdm





path = 'Datasets/joy/' #ADD NAME OF FOLDER CREATED FROM CNN_datagen.py FOR CODE TO RUN CORRECTLY (replace joy with folder name)
filename = 'data.csv'
files = os.listdir(path)
dim = (100, 100)
cls = 1
df = pd.DataFrame(columns = [f'pix-{i}' for i in range(1, 1+(dim[0]*dim[1]))]+['class'])
for i in tqdm(range(1, 1+len(files))):
    img =Image.open(path+files[i-1])
    df.loc[i] = list(img.getdata()) + [cls]

df.to_csv(filename,index = False)
print('Task Completed')

df = pd.read_csv('data.csv', index_col=0)

X = df.iloc[:, :100*100].values.reshape(-1, 100, 100, 1) 
y = df.iloc[:, -1].values
print(X.shape, y.shape)

y = to_categorical(y, num_classes= 1+ df.loc[:, 'class'].unique().shape[0])


X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=42, test_size=0.15)
print(f'Train Size - {X_train.shape}\nTest Size - {X_test.shape}')

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=10,
                                   width_shift_range=0.25,
                                   height_shift_range=0.25,
                                   shear_range=0.1,
                                   zoom_range=0.25,
                                   horizontal_flip=False)

valid_datagen = ImageDataGenerator(rescale=1./255.)

num_classes = 4
model_name = 'Face_trained_model_'+datetime.now().strftime("%H_%M_%S_")

model = Sequential(name = model_name)

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(100, 100, 1)))
model.add(BatchNormalization()) #----------------
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization()) #----------------
model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(BatchNormalization()) #----------------
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2)) #----------------

model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))
#model.summary()

learning_rate = 0.001
optimizer = RMSprop(lr=learning_rate)

model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'], run_eagerly=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=200,
                                            verbose=1,
                                            factor=0.2)

ch = ModelCheckpoint('models/'+model_name+'.h5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=200)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/"+datetime.now().strftime("%Y%m%d-%H%M%S"))

epochs = 50
batch_size = 256

history = model.fit(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                              steps_per_epoch= X_train.shape[0]//batch_size,
                              epochs=epochs,
                              validation_data=valid_datagen.flow(X_test, y_test),
                              validation_steps=50,
                              verbose = 1,
                              callbacks=[learning_rate_reduction, es, ch, tensorboard_callback])

loss, acc = model.evaluate(valid_datagen.flow(X_test, y_test)) 

print(f'Loss: {loss}\nAccuracy: {acc*100}')

# Plot training & test accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy'], loc='upper left')
plt.show()

# Plot training & test loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['loss'], loc='upper left')
plt.show()

