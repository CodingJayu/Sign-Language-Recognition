import tensorflow as tf
from tensorflow import keras
import numpy as np
from cvzone.HandTrackingModule import HandDetector

model = keras.models.load_model('marathi_module/Models/Trained_model.h5',compile=False)
img_height=300
img_width=300
class_names=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

marathi={1:'क',2:'ख',3:'ग',4:'घ',5:'च',6:'छ',7:'ज',8:'झ',9:'त्र',10:'ट',11:'ठ',12:'ड',13:'ढ',14:'ण',15:'त',16:'थ',17:'द',18:'ध',19:'न',20:'प',21:'फ',22:'ब',23:'भ',24:'म',25:'य',26:'र',27:'ल',28:'व',29:'श',30:'ष',31:'स',32:'ह',33:'ळ',34:'क्ष',35:'ज्ञ'}

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
