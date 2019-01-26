import keras
from keras.preprocessing import image
from keras import backend as K

import cv2
import numpy as np


# load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def casscasde_image(img):
   
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # send face through model, with scaleFactor and minNeighbors arguments
    faces = face_cascade.detectMultiScale(img, 1.4, 5)

    # draw rectangle around the face detected by the model, then crop it
    for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    crop_img = img[y:y+h, x:x+w]

    # resize cropped face to 48 by 48 pixels
    pixel_img = cv2.resize(crop_img, (48,48))



# scale pixels from 0 to 1 (originally from 0 to 255)
def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

processed = preprocess_input(pixel_img)

# reshape for model
gray_face = np.expand_dims(processed, 0)

# reshape for model
gray_face = np.expand_dims(gray_face, -1)

# load emotion model
from keras.models import load_model
emotion_classifier = load_model('fer2013_mini_XCEPTION.119-0.65.hdf5')

# predict with emotion model
prediction = emotion_classifier.predict(gray_face)

# check below for emotion label
result = np.argmax(prediction)

# emotion_labels = {
#     0: 'angry',
#      1: 'disgust',
#      2: 'fear',
#      3: 'happy',
#      4: 'sad',
#      5: 'surprise',
#      6: 'neutral'
# }


def findlabel(result):
    if result == 0:
        final_prediction = 'angry'
    elif result == 1:
        final_prediction = 'disgust'
    elif result == 2:
        final_prediction = 'fear'
    elif result == 3:
        final_prediction = 'happy'
    elif result == 4:
        final_prediction = 'sad'
    elif result == 5:
        final_prediction = 'surprise'
    else:
        final_prediction = 'neutral'
    return final_prediction

