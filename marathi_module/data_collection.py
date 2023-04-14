import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import string

base_folder = "marathi_module/Data"

marathi=['क','ख','ग','घ','च','छ','ज','झ','त्र','ट','ठ','ड','ढ','ण','त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह','ळ','क्ष','ज्ञ']

#Create Data Set Folder
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
if not os.path.exists(f"{base_folder}/train"):
    os.makedirs(f"{base_folder}/train")
if not os.path.exists(f"{base_folder}/test"):
    os.makedirs(f"{base_folder}/test")
for i in marathi:
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
             'क': len(os.listdir(folder+"/क")),
             'ख': len(os.listdir(folder+"/ख")),
             'ग': len(os.listdir(folder+"/ग")),
             'घ': len(os.listdir(folder+"/घ")),
             'च': len(os.listdir(folder+"/च")),
             'छ': len(os.listdir(folder+"/छ")),
             'ज': len(os.listdir(folder+"/ज")),
             'झ': len(os.listdir(folder+"/झ")),
             'त्र': len(os.listdir(folder+"/त्र")),
             'ट': len(os.listdir(folder+"/ट")),
             'ठ': len(os.listdir(folder+"/ठ")),
             'ड': len(os.listdir(folder+"/ड")),
             'ढ': len(os.listdir(folder+"/ढ")),
             'ण': len(os.listdir(folder+"/ण")),
             'त': len(os.listdir(folder+"/त")),
             'थ': len(os.listdir(folder+"/थ")),
             'द': len(os.listdir(folder+"/द")),
             'ध': len(os.listdir(folder+"/ध")),
             'न': len(os.listdir(folder+"/न")),
             'प': len(os.listdir(folder+"/प")),
             'फ': len(os.listdir(folder+"/फ")),
             'ब': len(os.listdir(folder+"/ब")),
             'भ': len(os.listdir(folder+"/भ")),
             'म': len(os.listdir(folder+"/म")),
             'य': len(os.listdir(folder+"/य")),
             'र': len(os.listdir(folder+"/र")),
             'ल': len(os.listdir(folder+"/ल")),
             'व': len(os.listdir(folder+"/व")),
             'श': len(os.listdir(folder+"/श")),
             'ष': len(os.listdir(folder+"/ष")),
             'स': len(os.listdir(folder+"/स")),
             'ह': len(os.listdir(folder+"/ह")),
             'ळ': len(os.listdir(folder+"/ळ")),
             'क्ष': len(os.listdir(folder+"/क्ष")),
             'ज्ञ': len(os.listdir(folder+"/ज्ञ"))
             }
    
    # Printing the count in each set to the screen

    # cv2.putText(img, "क : "+str(count['क']), (10, 17), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ख : "+str(count['ख']), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ग : "+str(count['ग']), (10, 43), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "घ : "+str(count['घ']), (10, 56), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "च : "+str(count['च']), (10, 69), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "छ : "+str(count['छ']), (10, 82), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ज : "+str(count['ज']), (10, 95), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "झ : "+str(count['झ']), (10, 108), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "त्र : "+str(count['त्र']), (10, 121), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ट : "+str(count['ट']), (10, 134), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ठ : "+str(count['ठ']), (10, 147), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ड : "+str(count['ड']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ढ : "+str(count['ढ']), (10, 173), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ण : "+str(count['ण']), (10, 186), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "त : "+str(count['त']), (10, 199), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "थ : "+str(count['थ']), (10, 212), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "द : "+str(count['द']), (10, 225), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ध : "+str(count['ध']), (10, 238), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "न : "+str(count['न']), (10, 251), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "प : "+str(count['प']), (10, 264), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "फ : "+str(count['फ']), (10, 277), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ब : "+str(count['ब']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "भ : "+str(count['भ']), (10, 303), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "म : "+str(count['म']), (10, 316), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "य : "+str(count['य']), (10, 329), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "र : "+str(count['र']), (10, 342), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ल : "+str(count['ल']), (10, 355), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "व : "+str(count['व']), (10, 368), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "श : "+str(count['श']), (10, 381), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ष : "+str(count['ष']), (10, 394), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "स : "+str(count['स']), (10, 407), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ह : "+str(count['ह']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ळ : "+str(count['ळ']), (10, 433), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "क्ष : "+str(count['क्ष']), (10, 446), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    # cv2.putText(img, "ज्ञ : "+str(count['ज्ञ']), (10, 459), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

    
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
        cv2.imwrite(f'{folder}/क/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("b"):
        cv2.imwrite(f'{folder}/ख/Image_{time.time()}.jpg',imgWhite)
    
    if key == ord("c"):
        cv2.imwrite(f'{folder}/ग/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("d"):
        cv2.imwrite(f'{folder}/घ/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("e"):
        cv2.imwrite(f'{folder}/च/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("f"):
        cv2.imwrite(f'{folder}/छ/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("g"):
        cv2.imwrite(f'{folder}/ज/Image_{time.time()}.jpg',imgWhite)

    if key == ord("h"):
        cv2.imwrite(f'{folder}/झ/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("i"):
        cv2.imwrite(f'{folder}/त्र/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("j"):
        cv2.imwrite(f'{folder}/ट/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("k"):
        cv2.imwrite(f'{folder}/ठ/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("l"):
        cv2.imwrite(f'{folder}/ड/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("m"):
        cv2.imwrite(f'{folder}/ढ/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("n"):
        cv2.imwrite(f'{folder}/ण/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("o"):
        cv2.imwrite(f'{folder}/त/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("p"):
        cv2.imwrite(f'{folder}/थ/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("q"):
        cv2.imwrite(f'{folder}/द/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("r"):
        cv2.imwrite(f'{folder}/ध/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("s"):
        cv2.imwrite(f'{folder}/न/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("t"):
        cv2.imwrite(f'{folder}/प/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("u"):
        cv2.imwrite(f'{folder}/फ/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("v"):
        cv2.imwrite(f'{folder}/ब/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("w"):
        cv2.imwrite(f'{folder}/भ/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("x"):
        cv2.imwrite(f'{folder}/म/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("y"):
        cv2.imwrite(f'{folder}/य/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("z"):
        cv2.imwrite(f'{folder}/र/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("1"):
        cv2.imwrite(f'{folder}/ल/Image_{time.time()}.jpg',imgWhite)
     
    if key == ord("2"):
        cv2.imwrite(f'{folder}/व/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("3"):
        cv2.imwrite(f'{folder}/श/Image_{time.time()}.jpg',imgWhite)
       
    if key == ord("4"):
        cv2.imwrite(f'{folder}/ष/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("5"):
        cv2.imwrite(f'{folder}/स/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("6"):
        cv2.imwrite(f'{folder}/ह/Image_{time.time()}.jpg',imgWhite)
        
    if key == ord("7"):
        cv2.imwrite(f'{folder}/ळ/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("8"):
        cv2.imwrite(f'{folder}/क्ष/Image_{time.time()}.jpg',imgWhite)
      
    if key == ord("9"):
        cv2.imwrite(f'{folder}/ज्ञ/Image_{time.time()}.jpg',imgWhite)


