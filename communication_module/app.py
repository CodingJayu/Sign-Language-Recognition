import tensorflow as tf
from tensorflow import keras
import numpy as np


model = keras.models.load_model('english_module/Models/Trained_model.h5',compile=False)
img_height=300
img_width=300
class_names=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def output(img):
    
    # img = tf.keras.utils.load_img(
    #     "a.jpg", target_size=(img_height, img_width)
    # )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )

    return class_names[np.argmax(score)],np.argmax(score)
