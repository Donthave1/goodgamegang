import os
from flask import Flask, jsonify, render_template, request, redirect

import keras
from keras.preprocessing import image
from keras import backend as K
from keras.models import load_model

from statistics import mode

import cv2
import numpy as np

from function import detect_faces
from function import apply_offsets
from function import load_detection_model
from function import preprocess_input
from function import findlabel


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

detection_model = None
emotion_model = None
graph = None

# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
def load_model():
    global detection_model
    global emotion_model
    global graph
    detection_model = load_detection_model("model/haarcascade_frontalface_default.xml")
    emotion_model = keras.models.load_model("model/fer2013_mini_XCEPTION.119-0.65.hdf5", compile=False)
    graph = K.get_session().graph

load_model()

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
no_face = None

def prepare_image(img):
    global no_face

    gray_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # gray_image = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(detection_model, gray_image)

    for face_coordinates in faces:
        print("@@@@@@@@@@@@@@@@@@@@ face coords 57")
        print(face_coordinates)
        print("@@@@@@@@@@@@@@@@@@@@ face coords 59")

        try:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            gray_face = cv2.resize(gray_face, (48,48))
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
        except ValueError:
            print('No Face 73')

    try:
        no_face = False
        return gray_face

    except:
        print("******No face detected 73")
        no_face = True
        print(no_face)
        print("******No face detected 75")
        return

@app.route('/')
def index_page():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    data = {"success": "No face detected"}
    print("does it get here 86")
    print(no_face)

    if no_face is True:
        return jsonify(data)

    if request.method == 'POST':

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)

            # Load the saved image using Keras and resize it to the mnist
            # format of 48x48 pixels
            # image_size = (48, 48)
            im = image.load_img(filepath, grayscale=True)

            # Convert the 2D image to an array of pixel values
            image_array = prepare_image(filepath)

            # Get the tensorflow default graph and use it to make predictions
            global graph
            if no_face is False:
                with graph.as_default():

                    # Use the model to make a prediction
                    predicted_digit = emotion_model.predict(image_array)[0]
                    data["probabilities of all outcomes"] = str(predicted_digit)

                    emotion_label_arg = np.argmax(predicted_digit)
                    data["final_prediction"] = str(findlabel(emotion_label_arg))

                    # indicate that the request was a success
                    data["success"] = True
                
            return jsonify(data)



if __name__ == "__main__":
    app.run(debug=True)
