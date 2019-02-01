import os
from flask import Flask, request, jsonify

import keras
from keras.preprocessing import image
from keras import backend as K
from keras.models import load_model

from statistics import mode

import cv2
import numpy as np

from functions import detect_faces
from functions import apply_offsets
from functions import load_detection_model
from functions import preprocess_input


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
    detection_model = load_detection_model("models/haarcascade_frontalface_default.xml")
    emotion_model = keras.models.load_model("models/fer2013_mini_XCEPTION.119-0.65.hdf5", compile=False)
    graph = K.get_session().graph

load_model()

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)

def prepare_image(img):
    emotion_target_size = emotion_model.input_shape[1:3]

    gray_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # gray_image = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(detection_model, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (48,48))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

    return gray_face


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        print(request)

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
            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
            global graph
            with graph.as_default():

                # Use the model to make a prediction
                predicted_digit = emotion_model.predict(image_array)[0]
                data["prediction"] = str(predicted_digit)

                emotion_label_arg = np.argmax(predicted_digit)
                data["final_prediction"] = str(findlabel(emotion_label_arg))
                


                # indicate that the request was a success
                data["success"] = True
                
            return jsonify(data)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)
