import requests
import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
import imghdr
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# Use query to sort out the data
for files in os.listdir("Unfiltered_Images"):
    print(sys.argv[1])
    query = requests.get(url = f"{sys.argv[1]}{files}")

    if query.json()["result"] == "Face":
        print("Got Face")
        Path(f"Unfiltered_Images/{files}").rename(f"Images/Face/{files}")
    else:
        print("Not Face")
        Path(f"Unfiltered_Images/{files}").rename(f"Images/Not_Face/{files}")

# Train classifer based on query
data_dir = "Images"
image_exts = ['jpeg','jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))

data = tf.keras.utils.image_dataset_from_directory('Images')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
data = data.map(lambda x,y: (x/255, y))
train_size = int(len(data)*.6)
val_size = int(len(data)*.2)
test_size = int(len(data)*.2)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# Create Simple CNN Model
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Model settings
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Model Training
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result(), re.result(), acc.result())

# Save Model
model.save(os.path.join('models','imageclassifier.h5'))