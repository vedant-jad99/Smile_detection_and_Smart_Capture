import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Dropout

#collecting and processing data
data = []; labels = [] #X and y
path = os.path.join(os.getcwd(), "dataset/positives")
for root, directory, files in os.walk(path):
    for filenames in files:
        current_path = os.path.join(path, filenames)
        try:
            img = cv2.imread(current_path)
            transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            transformed_img = np.reshape(transformed_img, (64, 64, 1))
            # if 'positives' in current_path.split('/'):
            data.append(transformed_img)
            labels.append(1)
            # elif 'negatives' in current_path.split('/'):
            #     data.append(transformed_img)
            #     labels.append(0)
        except:
            continue

path = os.path.join(os.getcwd(), "dataset/negatives")
for root, directory, files in os.walk(path):
    for filenames in files[:5000]:
        current_path = os.path.join(path, filenames)
        try:
            img = cv2.imread(current_path)
            transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            transformed_img = np.reshape(transformed_img, (64, 64, 1))
            # if 'positives' in current_path.split('/'):
            data.append(transformed_img)
            labels.append(0)
            # elif 'negatives' in current_path.split('/'):
            #     data.append(transformed_img)
            #     labels.append(0)
        except:
            continue

data = np.array(data)/255.0
labels = np_utils.to_categorical(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
print("Building model")
model = Sequential()
model.add(Conv2D(64, (3,3), (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(128, (5,5), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
print("training model")
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=10)
model.summary()
model.save("smile_detect")