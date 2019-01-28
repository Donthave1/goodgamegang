import os
try:
	os.chdir(os.path.join(os.getcwd(), 'static'))
	print(os.getcwd())
except:
	pass

from function import *
from keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import pprint as pp

emotion_model_path = 'models/fer2013_mini_XCEPTION.119-0.65.hdf5'
cascade_model_path = 'models/haarcascade_frontalface_default.xml'

emotion_classifier = load_model(emotion_model_path)
face_classifier = cv2.CascadeClassifier(cascade_model_path)

emotion_target_size = emotion_classifier.input_shape[1:3]

emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

cv2.namedWindow('frame')
video = cv2.VideoCapture(0)

emotion_offsets = (20, 40)

emotion_window = []

while(True):
    # Capture frame-by-frame
    ret, frame = video.read()

    # Our operations on the frame come here
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for face_coords in faces:
        
        x1, x2, y1, y2 = apply_offsets(face_coords, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
            
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if emotion_text == 'angry':
            color = [255, 0, 0]
        elif emotion_text == 'disgust':
            color = [128, 0, 128]
        elif emotion_text == 'fear':
            color = [255, 255, 0]
        elif emotion_text == 'happy':
            color = [255, 192, 203]
        elif emotion_text == 'sad':
            color = [0, 0, 255]
        elif emotion_text == 'surprise':
            color = [0, 0, 0]
        else:
            color = [255, 255, 255]

        draw_bounding_box(face_coords, rgb_image, color)
        draw_text(face_coords, rgb_image, emotion_text, color)
    
    # Display the resulting frame
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break