import tensorflow as tf
from tensorflow import keras
import numpy as np
from cvzone.HandTrackingModule import HandDetector

model = keras.models.load_model('marathi_module/Models/Trained_model.h5',compile=False)
img_height=300
img_width=300
class_names=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

marathi={1:'क',12:'ख',23:'ग',30:'घ',31:'च',5:'छ',33:'ज',34:'झ',27:'त्र',17:'ट',3:'ठ',4:'ड',2:'ढ',6:'ण',7:'त',8:'थ',9:'द',10:'ध',13:'न',11:'प',24:'फ',15:'ब',16:'भ',28:'म',18:'य',19:'र',20:'ल',21:'व',29:'श',32:'ष',25:'स',26:'ह',14:'ळ',22:'क्ष',35:'ज्ञ'}

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

    return marathi[class_names[np.argmax(score)]],np.argmax(score)