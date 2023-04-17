import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import math
from cvzone.HandTrackingModule import HandDetector

model = keras.models.load_model('marathi_module/Models/Trained_model.h5',compile=False)
img_height=300
img_width=300
class_names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','34']

marathi={1:'क',2:'ख',3:'ग',4:'घ',5:'च',6:'छ',7:'ज',8:'झ',9:'त्र',10:'ट',11:'ठ',12:'ड',13:'ढ',14:'ण',15:'त',16:'थ',17:'द',18:'ध',19:'न',20:'प',21:'फ',22:'ब',23:'भ',24:'म',25:'य',26:'र',27:'ल',28:'व',29:'श',30:'ष',31:'स',32:'ह',33:'ळ',34:'क्ष',35:'ज्ञ'}

def output(img):
    
    # img = tf.keras.utils.load_img(
    #     "a.jpg", target_size=(img_height, img_width)
    # )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(marathi[class_names[np.argmax(score)]], 100 * np.max(score))
    )

    return class_names[np.argmax(score)]


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
 
offset = 20
imgSize = 300
 
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
 
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
 
        imgCropShape = imgCrop.shape
 
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
    
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            # cv2.imshow("Image", img)
            # imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            cv2.imshow("ImageWhite", imgWhite)
            output(imgWhite)
        
        except Exception as e:
            # By this way we can know about the type of error occurring
            print("The error is: ",e)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    #System Exit Key
    if key == 27:
        break
    