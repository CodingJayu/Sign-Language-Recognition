import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

base_folder = "marathi_module/Data"

marathi={1:'क',2:'ख',3:'ग',4:'घ',5:'च',6:'छ',7:'ज',8:'झ',9:'त्र',10:'ट',11:'ठ',12:'ड',13:'ढ',14:'ण',15:'त',16:'थ',17:'द',18:'ध',19:'न',20:'प',21:'फ',22:'ब',23:'भ',24:'म',25:'य',26:'र',27:'ल',28:'व',29:'श',30:'ष',31:'स',32:'ह',33:'ळ',34:'क्ष',35:'ज्ञ'}

#Create Data Set Folder
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
if not os.path.exists(f"{base_folder}/train"):
    os.makedirs(f"{base_folder}/train")
if not os.path.exists(f"{base_folder}/test"):
    os.makedirs(f"{base_folder}/test")
for i in range(36):
    if not os.path.exists(f"{base_folder}/train/" +str(i)):
        os.makedirs(f"{base_folder}/train/"+str(i))
    if not os.path.exists(f"{base_folder}/test/" + str(i)):
        os.makedirs(f"{base_folder}/test/"+str(i))


 
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
             '1': len(os.listdir(folder+"/1")),
             '2': len(os.listdir(folder+"/2")),
             '3': len(os.listdir(folder+"/3")),
             '4': len(os.listdir(folder+"/4")),
             '5': len(os.listdir(folder+"/5")),
             '6': len(os.listdir(folder+"/6")),
             '7': len(os.listdir(folder+"/7")),
             '8': len(os.listdir(folder+"/8")),
             '9': len(os.listdir(folder+"/9")),
             '10': len(os.listdir(folder+"/10")),
             '11': len(os.listdir(folder+"/11")),
             '12': len(os.listdir(folder+"/12")),
             '13': len(os.listdir(folder+"/13")),
             '14': len(os.listdir(folder+"/14")),
             '15': len(os.listdir(folder+"/15")),
             '16': len(os.listdir(folder+"/16")),
             '17': len(os.listdir(folder+"/17")),
             '18': len(os.listdir(folder+"/18")),
             '19': len(os.listdir(folder+"/19")),
             '20': len(os.listdir(folder+"/20")),
             '21': len(os.listdir(folder+"/21")),
             '22': len(os.listdir(folder+"/22")),
             '23': len(os.listdir(folder+"/23")),
             '24': len(os.listdir(folder+"/24")),
             '25': len(os.listdir(folder+"/25")),
             '26': len(os.listdir(folder+"/26")),
             '27': len(os.listdir(folder+"/27")),
             '28': len(os.listdir(folder+"/28")),
             '29': len(os.listdir(folder+"/29")),
             '30': len(os.listdir(folder+"/30")),
             '31': len(os.listdir(folder+"/31")),
             '32': len(os.listdir(folder+"/32")),
             '33': len(os.listdir(folder+"/33")),
             '34': len(os.listdir(folder+"/34")),
             '35': len(os.listdir(folder+"/35"))
             }
    
    # Printing the count in each set to the screen

    cv2.putText(img, "1 : "+str(count['1']), (10, 17), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "2 : "+str(count['2']), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "3 : "+str(count['3']), (10, 43), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "4 : "+str(count['4']), (10, 56), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "5 : "+str(count['5']), (10, 69), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "6 : "+str(count['6']), (10, 82), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "7 : "+str(count['7']), (10, 95), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "8 : "+str(count['8']), (10, 108), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "9 : "+str(count['9']), (10, 121), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "10 : "+str(count['10']), (10, 134), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "11 : "+str(count['11']), (10, 147), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "12 : "+str(count['12']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "13 : "+str(count['13']), (10, 173), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "14 : "+str(count['14']), (10, 186), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "15 : "+str(count['15']), (10, 199), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "16 : "+str(count['16']), (10, 212), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "17 : "+str(count['17']), (10, 225), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "18 : "+str(count['18']), (10, 238), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "19 : "+str(count['19']), (10, 251), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "20 : "+str(count['20']), (10, 264), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "21 : "+str(count['21']), (10, 277), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "22 : "+str(count['22']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "23 : "+str(count['23']), (10, 303), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "24 : "+str(count['24']), (10, 316), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "25 : "+str(count['25']), (10, 329), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "26 : "+str(count['26']), (10, 342), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "27 : "+str(count['27']), (10, 355), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "28 : "+str(count['28']), (10, 368), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "29 : "+str(count['29']), (10, 381), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "30 : "+str(count['30']), (10, 394), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "31 : "+str(count['31']), (10, 407), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "32 : "+str(count['32']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "33 : "+str(count['33']), (10, 433), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "34 : "+str(count['34']), (10, 446), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(img, "35 : "+str(count['35']), (10, 459), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

    
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

            # Color to Grey Image
            # imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            # imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
    
            cv2.imshow("ImageCrop", imgCrop)
            cv2.moveWindow("ImageCrop",700,0) # WINDOW POSITION
            # cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            # By this way we can know about the type of error occurring
            print("The error is: ",e)
    
    cv2.imshow("Image", img)
    cv2.moveWindow("Image",20,0)    # WINDOW POSITION
    key = cv2.waitKey(1)

    #System Exit Key
    if key == 27:
        break

    #Stores Alphabets in folder
    if key == ord("a"):
        cv2.imwrite(f'{folder}/1/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("b"):
        cv2.imwrite(f'{folder}/2/Image_{time.time()}.jpg',imgWhite)
    
    if key == ord("c"):
        cv2.imwrite(f'{folder}/3/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("d"):
        cv2.imwrite(f'{folder}/4/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("e"):
        cv2.imwrite(f'{folder}/5/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("f"):
        cv2.imwrite(f'{folder}/6/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("g"):
        cv2.imwrite(f'{folder}/7/Image_{time.time()}.jpg',imgWhite)

    if key == ord("h"):
        cv2.imwrite(f'{folder}/8/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("i"):
        cv2.imwrite(f'{folder}/9/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("j"):
        cv2.imwrite(f'{folder}/10/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("k"):
        cv2.imwrite(f'{folder}/11/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("l"):
        cv2.imwrite(f'{folder}/12/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("m"):
        cv2.imwrite(f'{folder}/13/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("n"):
        cv2.imwrite(f'{folder}/14/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("o"):
        cv2.imwrite(f'{folder}/15/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("p"):
        cv2.imwrite(f'{folder}/16/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("q"):
        cv2.imwrite(f'{folder}/17/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("r"):
        cv2.imwrite(f'{folder}/18/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("s"):
        cv2.imwrite(f'{folder}/19/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("t"):
        cv2.imwrite(f'{folder}/20/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("u"):
        cv2.imwrite(f'{folder}/21/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("v"):
        cv2.imwrite(f'{folder}/22/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("w"):
        cv2.imwrite(f'{folder}/23/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("x"):
        cv2.imwrite(f'{folder}/24/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("y"):
        cv2.imwrite(f'{folder}/25/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("z"):
        cv2.imwrite(f'{folder}/26/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("1"):
        cv2.imwrite(f'{folder}/27/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("2"):
        cv2.imwrite(f'{folder}/28/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("3"):
        cv2.imwrite(f'{folder}/29/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("4"):
        cv2.imwrite(f'{folder}/30/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("5"):
        cv2.imwrite(f'{folder}/31/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("6"):
        cv2.imwrite(f'{folder}/32/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("7"):
        cv2.imwrite(f'{folder}/33/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("8"):
        cv2.imwrite(f'{folder}/34/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("9"):
        cv2.imwrite(f'{folder}/35/Image_{time.time()}.jpg',imgWhite)

#marathi data collection model ends 
