import tensorflow as tf
from tensorflow import keras

import cv2
import numpy as np

model = keras.models.load_model('trained.h5')

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

def load_image(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(150, 150)) 
    img_tensor = keras.preprocessing.image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

img = load_image('6.jpg')
classes = model.predict_classes(img)

print(classes)