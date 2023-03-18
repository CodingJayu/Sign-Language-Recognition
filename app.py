import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import math
from cvzone.HandTrackingModule import HandDetector

model = keras.models.load_model('Trained_model.h5',compile=False)
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

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
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
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    #System Exit Key
    if key == 27:
        break
