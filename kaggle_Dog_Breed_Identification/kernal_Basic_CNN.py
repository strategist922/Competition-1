import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import os
from shutil import copyfile,copy,copy2
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D,Dropout
from keras.preprocessing.image import ImageDataGenerator

labels = pd.read_csv('../input/labels.csv')

labels_dict = {i:j for i,j in zip(labels['id'],labels['breed'])}
classes = set(labels_dict.values())
images = [f for f in os.listdir('../input/train')]
#(images)
#os.makedirs('training_images')
#os.makedirs('validation_images')

if not os.path.exists('training_images'):
    os.makedirs('training_images')

if  not os.path.exists('validation_images'):
    os.makedirs('validation_images')
from os.path import join


for item in images:
    filekey = os.path.splitext(item)[0]
    if not os.path.exists('training_images/'+labels_dict[filekey]):
        os.makedirs('training_images/'+labels_dict[filekey])
    if not os.path.exists('validation_images/'+labels_dict[filekey]):
        os.makedirs('validation_images/'+labels_dict[filekey])

count = 0 
destination_directory = 'training_images'
cwd = os.getcwd()
print(cwd)
for item in images:
    if count >7999:
        destination_directory = 'validation_images'
    filekey = os.path.splitext(item)[0]
    dest_file_path = join(destination_directory,labels_dict[filekey],item)
    src_file_path = join("..","input","train",item)
    #print(src_file_path)
    if not os.path.exists(dest_file_path):
        copyfile(src_file_path, dest_file_path)
    count +=1
    

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_images',
        target_size=(128, 128),
        batch_size=20,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'validation_images',
        target_size=(128, 128),
        batch_size=20,
        class_mode='categorical')
        
from keras.layers import Dropout
clf = Sequential()
#Convolution
#32 is number of kernals of 3x3, we can use 64 128 256 etc in next layers
#input shape can be 128, 256 later
clf.add(Conv2D(32,(3,3),input_shape=(128,128,3),activation='relu'))
#Max Pooling size reduces divided by 2
clf.add(MaxPooling2D(pool_size=(2,2)))      


#clf.add(Dropout(0.5))

clf.add(Conv2D(32,(3,3), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))
#clf.add(Dropout(0.25))

clf.add(Conv2D(64, (3, 3), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2, 2)))
#clf.add(Dropout(0.10))
#Flattening
clf.add(Flatten())
        
#Adding An ANN
#lets take 128 hidden nodes in hidden layer
#clf.add(Dense(units=128,activation='relu'))
clf.add(Dense(units=64, activation='relu'))
clf.add(Dropout(0.5))
clf.add(Dense(units=120,activation='softmax'))
#stochastic gradient descent -Adam -optimizer
#loss func categorical cross entropy
#metrics = accuracy
clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=6)

hist=clf.fit_generator(
        training_set,
        steps_per_epoch=400,
        epochs=50,
        validation_data=test_set,
        validation_steps=2222)
#callbacks=[early_stopping_monitor])

import cv2
test_set = []
test_set_ids = []
for curImage in os.listdir('../input/test'):
    test_set_ids.append(os.path.splitext(curImage)[0])
    #print(os.path.splitext(curImage)[0])
    curImage = cv2.imread('../input/test/'+curImage)
    test_set.append(cv2.resize(curImage,(128, 128)))
    
test_set = np.array(test_set, np.float32)/255.0
predictions= clf.predict(test_set)

classes= {index:breed for breed,index in training_set.class_indices.items()}
column_names = [classes[i] for i in range(120)]
predictions_df = pd.DataFrame(predictions)
predictions_df.columns = column_names
predictions_df.insert(0,'id', test_set_ids)
predictions_df.to_csv('interim_submission.csv',sep=",")

