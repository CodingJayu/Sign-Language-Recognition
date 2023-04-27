#Modified by Augmented Startups 2021
#Face Landmark User Interface with StreamLit
#Watch Computer Vision Tutorials at www.augmentedstartups.info/YouTube
import cv2
import math
import os
from cvzone.HandTrackingModule import HandDetector
import streamlit as st
import numpy as np
import english_module.app as eng_run
from mss import mss
# import marathi_module.app as mar_run
from screeninfo import get_monitors
import mediapipe as mp
from tensorflow import keras

mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_holistic = mp.solutions.holistic # Holistic model





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

    # st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Start Webcam')
    use_scr = st.sidebar.button('Start Screen Recorder')
  
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

    
    if use_scr:
        for m in get_monitors():
            width=m.width
            height=m.height

        bounding_box={'top':0,'left':0,'width':width,'height':height}
        sct=mss()
        detector = HandDetector(maxHands=1)
        offset = 20
        imgSize = 300
    
        
        while True:

            img=np.array(sct.grab(bounding_box))
            img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
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
                    kpi2_text.write(f"<h6 style='text-align: center; color: red;'>Hand Out of Frame</h6>", unsafe_allow_html=True)
    
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

    use_webcam = st.sidebar.button('Start Webcam')
    use_scr = st.sidebar.button('Start Screen Recorder')
  
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


    if use_scr:
        for m in get_monitors():
            width=m.width
            height=m.height

        bounding_box={'top':0,'left':0,'width':width,'height':height}
        sct=mss()

        detector = HandDetector(maxHands=1)
        offset = 20
        imgSize = 300
    
        
        while True:

            img=np.array(sct.grab(bounding_box))
            img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
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
                    result,acc=mar_run.output(imgWhite)
                    kpi1_text.write(f"<h6 style='text-align: center; color: green;'>{result}</h6>", unsafe_allow_html=True)
                    kpi2_text.write(f"<h6 style='text-align: center; color: green;'>OK</h6>", unsafe_allow_html=True)
                
                
                except Exception as e:
                    # By this way we can know about the type of error occurring
                    kpi2_text.write(f"<h6 style='text-align: center; color: red;'>Hand Out of Frame</h6>", unsafe_allow_html=True)
    
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
                    result,acc=mar_run.output(imgWhite)
                    kpi1_text.write(f"<h6 style='text-align: center; color: green;'>{result}</h6>", unsafe_allow_html=True)
                    kpi2_text.write(f"<h6 style='text-align: center; color: green;'>OK</h6>", unsafe_allow_html=True)
                
                
                except Exception as e:
                    # By this way we can know about the type of error occurring
                    kpi2_text.write(f"<h6 style='text-align: center; color: red;'>Error:{e}</h6>", unsafe_allow_html=True)
           
  
 
elif app_mode =='Communication':
     # st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Start Webcam')
    # use_scr = st.sidebar.button('Start Screen Recorder')
  
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
        st.markdown("**Output English Sentence**")
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

        actions = np.array(['Hello', 'Thanks', 'I love you','sorry','Good Morning','Good Night','Good Afternoon','Happy','Man','Women','Have You Eaten ?','Bye','I','Your','You','Good','Very Good','Bad','All the Best','Yes','No'])

        colors = [(245,117,16), (117,245,16), (16,117,245)]

        def prob_viz(res, actions, input_frame, colors):
            output_frame = input_frame.copy()
            for num, prob in enumerate(res):
                cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
            return output_frame

        def mediapipe_detection(image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False                  # Image is no longer writeable
            results = model.process(image)                 # Make prediction
            image.flags.writeable = True                   # Image is now writeable 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
            return image, results

        def draw_styled_landmarks(image, results):
            # Draw face connections
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    ) 
            # Draw pose connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    ) 
            # Draw left hand connections
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    ) 
            # Draw right hand connections  
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    ) 
            
        def extract_keypoints(results):
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            return np.concatenate([pose, face, lh, rh])


        # Set mediapipe model 
                
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        model = keras.models.load_model('communication_module/Model/Communication.h5',compile=False)

        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                    
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                    

                frame = cv2.resize(image,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = width)
                stframe.image(frame,channels = 'BGR',use_column_width=True)

                kpi1_text.write(f"<h6 style='text-align: center; color: green;'>{' '.join(sentence)}</h6>", unsafe_allow_html=True)
                kpi2_text.write(f"<h6 style='text-align: center; color: green;'>OK</h6>", unsafe_allow_html=True)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()