import numpy as np
from scipy.misc import imread, imresize
from scipy.io import loadmat
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image


def get_labels(dataset):
    return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

def load_data(self):
        image_size = (48, 48)
        data = pd.read_csv('datasets/fer2013.csv')
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions





def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0, font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coords, offsets):
    x, y, width, height = face_coords
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


# def _imread(image_name):
#         return imread(image_name)


# def _imresize(image_array, size):
#         return imresize(image_array, size)


# def to_categorical(integer_classes, num_classes=2):
#     integer_classes = np.asarray(integer_classes, dtype='int')
#     num_samples = integer_classes.shape[0]
#     categorical = np.zeros((num_samples, num_classes))
#     categorical[np.arange(num_samples), integer_classes] = 1
#     return categorical
