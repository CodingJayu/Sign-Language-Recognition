import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import string

base_folder = "marathi_module/Data"

#Create Data Set Folder
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
if not os.path.exists(f"{base_folder}/train"):
    os.makedirs(f"{base_folder}/train")
if not os.path.exists(f"{base_folder}/test"):
    os.makedirs(f"{base_folder}/test")
for i in range(10):
    if not os.path.exists(f"{base_folder}/train/" + str(i)):
        os.makedirs(f"{base_folder}/train/"+str(i))
    if not os.path.exists(f"{base_folder}/test/" + str(i)):
        os.makedirs(f"{base_folder}/test/"+str(i))

for i in string.ascii_uppercase:
    if not os.path.exists(f"{base_folder}/train/" + i):
        os.makedirs(f"{base_folder}/train/"+i)
    if not os.path.exists(f"{base_folder}/test/" + i):
        os.makedirs(f"{base_folder}/test/"+i)

 
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
 
offset = 20
imgSize = 300

choice=int(input("Select the mode for training\n1.Training\n2.Test\n"))
if(choice==1):
    folder=base_folder+"/train" 
elif(choice==2):
    folder=base_folder+"/test" 

 
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    # Getting count of existing images
    count = {
             'zero': len(os.listdir(folder+"/0")),
             'one': len(os.listdir(folder+"/1")),
             'two': len(os.listdir(folder+"/2")),
             'three': len(os.listdir(folder+"/3")),
             'four': len(os.listdir(folder+"/4")),
             'five': len(os.listdir(folder+"/5")),
             'six': len(os.listdir(folder+"/6")),
             'seven': len(os.listdir(folder+"/7")),
             'eight': len(os.listdir(folder+"/8")),
             'nine': len(os.listdir(folder+"/9")),
             'a': len(os.listdir(folder+"/A")),
             'b': len(os.listdir(folder+"/B")),
             'c': len(os.listdir(folder+"/C")),
             'd': len(os.listdir(folder+"/D")),
             'e': len(os.listdir(folder+"/E")),
             'f': len(os.listdir(folder+"/F")),
             'g': len(os.listdir(folder+"/G")),
             'h': len(os.listdir(folder+"/H")),
             'i': len(os.listdir(folder+"/I")),
             'j': len(os.listdir(folder+"/J")),
             'k': len(os.listdir(folder+"/K")),
             'l': len(os.listdir(folder+"/L")),
             'm': len(os.listdir(folder+"/M")),
             'n': len(os.listdir(folder+"/N")),
             'o': len(os.listdir(folder+"/O")),
             'p': len(os.listdir(folder+"/P")),
             'q': len(os.listdir(folder+"/Q")),
             'r': len(os.listdir(folder+"/R")),
             's': len(os.listdir(folder+"/S")),
             't': len(os.listdir(folder+"/T")),
             'u': len(os.listdir(folder+"/U")),
             'v': len(os.listdir(folder+"/V")),
             'w': len(os.listdir(folder+"/W")),
             'x': len(os.listdir(folder+"/X")),
             'y': len(os.listdir(folder+"/Y")),
             'z': len(os.listdir(folder+"/Z"))
             }
    
    # Printing the count in each set to the screen

    cv2.putText(img, "ZERO : "+str(count['zero']), (10, 17), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "ONE : "+str(count['one']), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "TWO : "+str(count['two']), (10, 43), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "THREE : "+str(count['three']), (10, 56), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "FOUR : "+str(count['four']), (10, 69), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "FIVE : "+str(count['five']), (10, 82), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "SIX : "+str(count['six']), (10, 95), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "SEVEN : "+str(count['seven']), (10, 108), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "EIGHT : "+str(count['eight']), (10, 121), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "NINE : "+str(count['nine']), (10, 134), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "A : "+str(count['a']), (10, 147), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "B : "+str(count['b']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "C : "+str(count['c']), (10, 173), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "D : "+str(count['d']), (10, 186), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "E : "+str(count['e']), (10, 199), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "F : "+str(count['f']), (10, 212), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "G : "+str(count['g']), (10, 225), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "H : "+str(count['h']), (10, 238), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "I : "+str(count['i']), (10, 251), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "J : "+str(count['j']), (10, 264), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "K : "+str(count['k']), (10, 277), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "L : "+str(count['l']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "M : "+str(count['m']), (10, 303), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "N : "+str(count['n']), (10, 316), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "O : "+str(count['o']), (10, 329), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "P : "+str(count['p']), (10, 342), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "Q : "+str(count['q']), (10, 355), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "R : "+str(count['r']), (10, 368), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "S : "+str(count['s']), (10, 381), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "T : "+str(count['t']), (10, 394), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "U : "+str(count['u']), (10, 407), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "V : "+str(count['v']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "W : "+str(count['w']), (10, 433), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "X : "+str(count['x']), (10, 446), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "Y : "+str(count['y']), (10, 459), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "Z : "+str(count['z']), (10, 472), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    
    total=0
    for i in count:
        total=total+count[i]
    
    cv2.putText(img, "Total : "+str(total), (520, 17), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    

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

        # Color to Grey Image
        # imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
        # imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
 
        cv2.imshow("ImageCrop", imgCrop)
        cv2.moveWindow("ImageCrop",700,0) # WINDOW POSITION
        # cv2.imshow("ImageWhite", imgWhite)
       
        
    
    cv2.imshow("Image", img)
    cv2.moveWindow("Image",20,0)    # WINDOW POSITION
    key = cv2.waitKey(1)

    #System Exit Key
    if key == 27:
        break

    #Stores Alphabets in folder
    if key == ord("a"):
        cv2.imwrite(f'{folder}/A/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("b"):
        cv2.imwrite(f'{folder}/B/Image_{time.time()}.jpg',imgWhite)
    
    if key == ord("c"):
        cv2.imwrite(f'{folder}/C/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("d"):
        cv2.imwrite(f'{folder}/D/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("e"):
        cv2.imwrite(f'{folder}/E/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("f"):
        cv2.imwrite(f'{folder}/F/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("g"):
        cv2.imwrite(f'{folder}/G/Image_{time.time()}.jpg',imgWhite)

    if key == ord("h"):
        cv2.imwrite(f'{folder}/H/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("i"):
        cv2.imwrite(f'{folder}/I/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("j"):
        cv2.imwrite(f'{folder}/J/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("k"):
        cv2.imwrite(f'{folder}/K/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("l"):
        cv2.imwrite(f'{folder}/L/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("m"):
        cv2.imwrite(f'{folder}/M/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("n"):
        cv2.imwrite(f'{folder}/N/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("o"):
        cv2.imwrite(f'{folder}/O/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("p"):
        cv2.imwrite(f'{folder}/P/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("q"):
        cv2.imwrite(f'{folder}/Q/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("r"):
        cv2.imwrite(f'{folder}/R/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("s"):
        cv2.imwrite(f'{folder}/S/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("t"):
        cv2.imwrite(f'{folder}/T/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("u"):
        cv2.imwrite(f'{folder}/U/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("v"):
        cv2.imwrite(f'{folder}/V/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("w"):
        cv2.imwrite(f'{folder}/W/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("x"):
        cv2.imwrite(f'{folder}/X/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("y"):
        cv2.imwrite(f'{folder}/Y/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("z"):
        cv2.imwrite(f'{folder}/Z/Image_{time.time()}.jpg',imgWhite)
     

    #Numbers Stores
    if key == ord("0"):
        cv2.imwrite(f'{folder}/0/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("1"):
        cv2.imwrite(f'{folder}/1/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("2"):
        cv2.imwrite(f'{folder}/2/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("3"):
        cv2.imwrite(f'{folder}/3/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("4"):
        cv2.imwrite(f'{folder}/4/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("5"):
        cv2.imwrite(f'{folder}/5/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("6"):
        cv2.imwrite(f'{folder}/6/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("7"):
        cv2.imwrite(f'{folder}/7/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("8"):
        cv2.imwrite(f'{folder}/8/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("9"):
        cv2.imwrite(f'{folder}/9/Image_{time.time()}.jpg',imgWhite)

    
