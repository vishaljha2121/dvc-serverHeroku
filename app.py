# importing libraries
import os
import flask
import pandas as pd
import tensorflow as tf
from tensorflow import keras

keras.__version__
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# instantiate flask
app = flask.Flask(__name__)

# folder locations
STATIC_FOLDER = "static"
UPLOAD_FOLDER = STATIC_FOLDER + "/uploads"
MODEL_FOLDER = "/Models"


@app.route("/upload", methods=["POST"])
# file upload and saving
def upload():
    # fetch file from json input
    file = flask.request.files["image"]
    if file == "":
        return "Please enter an image"
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_location)

    # call prediction function
    result = predict_results(file_location)
    return result


# get image data
def get_image(file_location):
    image_loaded = image.load_img(file_location, target_size=(64, 64, 1))
    image_loaded = image.img_to_array(image_loaded)
    image_loaded = image_loaded / 255
    image_loaded = np.expand_dims(image_loaded, axis=0)
    return image_loaded


# generate predictions
def predict_results(data):
    image = get_image(data)
    prediction = model.predict(image, batch_size=None, steps=1)
    if prediction[:, :] > 0.5:
        label = "Dog"
    else:
        label = "Cat"
    return {"label": label}


def load__model():
    global model
    model = load_model(STATIC_FOLDER + MODEL_FOLDER + "/model.h5")
    global graph
    graph = tf.compat.v1.get_default_graph()


def start_server():
    app.run(debug=False)


def run():
    # load model
    print("[INFO] : Loading model.....")
    load__model()
    print("[INFO] : Model Loaded.")

    # run server
    print("[INFO] : Starting server")
    start_server()

    # fetch and save file -> load file -> predict
    " auto upload() call on post method to [/upload]"
    # show label
    # TODO


run()  # running all
