import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from IPython.display import display
from PIL import Image

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

classifier = Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape =(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(activation='relu',units=128))
classifier.add(Dense(activation='sigmoid',units=1))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing import image

#data augmentation

train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(
        'meme_d/memes',
        target_size=(64,64),
        batch_size=5,
        class_mode='binary'
        )

test_set=test_datagen.flow_from_directory(
        'meme_d/meme_test',
        target_size=(64,64),
        batch_size=5,
        class_mode='binary'
        )

classifier.fit_generator(
        training_set,
        steps_per_epoch=0.1,
        epochs=10,
        validation_data=test_set,
        validation_steps=10
        )

#saving model
classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")

while(1):
    x = input()
    test_image=image.load_img(x,target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=classifier.predict(test_image)
    training_set.class_indices
    if result[0][0]>=0.5:
        prediction='meme'
    else:
        prediction='not meme'
    print(prediction)
