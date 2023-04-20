#Modified by Augmented Startups 2021
#Face Landmark User Interface with StreamLit
#Watch Computer Vision Tutorials at www.augmentedstartups.info/YouTube
import cv2
import math
import os
from cvzone.HandTrackingModule import HandDetector
import streamlit as st
import cv2
import numpy as np
import english_module.app as eng_run
# import marathi_module.app as mar_run

st.title('Sign Language Recognition')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("Terminate"):
    st.success("You can Close Tab Now!")
    os.system("pkill streamlit")

st.sidebar.title('Sign Language Recognition')
st.sidebar.subheader('Parameters')

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
['English Language Letter','Marathi Language Letter','Communication']
)



if app_mode =='English Language Letter':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Start Webcam')
  
    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )

    kpi1, kpi2, kpi3= st.columns(3)

    with kpi1:
        st.markdown("**Output English Letter**")
        kpi1_text = st.markdown("None")
        
    with kpi2:
        st.markdown("**System Status**")
        kpi2_text = st.markdown("None")
    
    with kpi3:
        if st.button("Stop Capturing"):
            st.stop()
            
                
    

    st.markdown("<hr/>", unsafe_allow_html=True)

 
    st.markdown('## Input Video')

    stframe = st.empty()
    
    if use_webcam:
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        offset = 20
        imgSize = 300
        
        while True:
            flag=0
            success, img = cap.read()
            hands, img = detector.findHands(img)
            frame = cv2.resize(img,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = width)
            stframe.image(frame,channels = 'BGR',use_column_width=True)

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
                    # cv2.imshow("ImageWhite", imgWhite)
                    result,acc=eng_run.output(imgWhite)
                    kpi1_text.write(f"<h6 style='text-align: center; color: green;'>{result}</h6>", unsafe_allow_html=True)
                    kpi2_text.write(f"<h6 style='text-align: center; color: green;'>OK</h6>", unsafe_allow_html=True)
                
                
                except Exception as e:
                    # By this way we can know about the type of error occurring
                    kpi2_text.write(f"<h6 style='text-align: center; color: red;'>Error:{e}</h6>", unsafe_allow_html=True)
           
                
            
elif app_mode =='Marathi Language Letter':

    # st.set_option('deprecation.showfileUploaderEncoding', False)

    # use_webcam = st.sidebar.button('Start Webcam')
  
    # st.sidebar.markdown('---')
    # st.markdown(
    # """
    # <style>
    # [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    #     width: 400px;
    # }
    # [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    #     width: 400px;
    #     margin-left: -400px;
    # }
    # </style>
    # """,
    # unsafe_allow_html=True,
    #     )

    # kpi1, kpi2, kpi3= st.columns(3)

    # with kpi1:
    #     st.markdown("**Output English Letter**")
    #     kpi1_text = st.markdown("None")
        
    # with kpi2:
    #     st.markdown("**System Status**")
    #     kpi2_text = st.markdown("None")
    
    # with kpi3:
    #     if st.button("Stop Capturing"):
    #         st.stop()
            
                
    

    # st.markdown("<hr/>", unsafe_allow_html=True)

 
    # st.markdown('## Input Video')

    # stframe = st.empty()
    
    # if use_webcam:
    #     cap = cv2.VideoCapture(0)
    #     detector = HandDetector(maxHands=1)
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps_input = int(cap.get(cv2.CAP_PROP_FPS))
    #     offset = 20
    #     imgSize = 300
        
    #     while True:
    #         flag=0
    #         success, img = cap.read()
    #         hands, img = detector.findHands(img)
    #         frame = cv2.resize(img,(0,0),fx = 0.8 , fy = 0.8)
    #         frame = image_resize(image = frame, width = width)
    #         stframe.image(frame,channels = 'BGR',use_column_width=True)

    #         if hands:
    #             hand = hands[0]
    #             x, y, w, h = hand['bbox']
        
    #             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    #             imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        
    #             imgCropShape = imgCrop.shape
        
    #             aspectRatio = h / w

    #             try:
    #                 if aspectRatio > 1:
    #                     k = imgSize / h
    #                     wCal = math.ceil(k * w)
    #                     imgResize = cv2.resize(imgCrop, (wCal, imgSize))
    #                     imgResizeShape = imgResize.shape
    #                     wGap = math.ceil((imgSize - wCal) / 2)
    #                     imgWhite[:, wGap:wCal + wGap] = imgResize
            
    #                 else:
    #                     k = imgSize / w
    #                     hCal = math.ceil(k * h)
    #                     imgResize = cv2.resize(imgCrop, (imgSize, hCal))
    #                     imgResizeShape = imgResize.shape
    #                     hGap = math.ceil((imgSize - hCal) / 2)
    #                     imgWhite[hGap:hCal + hGap, :] = imgResize
                    
    #                 # cv2.imshow("Image", img)
    #                 # imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
    #                 # cv2.imshow("ImageWhite", imgWhite)
    #                 result,acc=mar_run.output(imgWhite)
    #                 kpi1_text.write(f"<h6 style='text-align: center; color: green;'>{result}</h6>", unsafe_allow_html=True)
    #                 kpi2_text.write(f"<h6 style='text-align: center; color: green;'>OK</h6>", unsafe_allow_html=True)
                
                
    #             except Exception as e:
    #                 # By this way we can know about the type of error occurring
    #                 kpi2_text.write(f"<h6 style='text-align: center; color: red;'>Hand Out of Frame</h6>", unsafe_allow_html=True)
    pass

elif app_mode =='Communication':
    #Code Writting Remaining
    pass