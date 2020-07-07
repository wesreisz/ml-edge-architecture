# USAGE
# Start the server:
# 	python app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils

from PIL import Image
import flask
import io
import requests

IMAGENET_WEIGHTS = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
imgnet_map = None

def load_imgnet_map():
    # get the json wordbag
    global imgnet_map
    response=requests.get(IMAGENET_WEIGHTS)
    imgnet_map=response.json()

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
    global model
    model=tf.keras.applications.xception.Xception(weights='imagenet',include_top=True)
    
def prepare_image(image):
    #make sure image is rgb
    if image.mode != "RGB":
        image = image.convert("RGB")
   
    #set image size to 299 x 299
    image = image.resize((299, 299))

    image=tf.keras.preprocessing.image.img_to_array(image)
    image=tf.keras.applications.xception.preprocess_input(image)
    return image

@app.route("/", methods=["GET"])
def index():
    return "<p>API Server</p>"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image)
            
            # make prediction
            predictions=model.predict(np.array([image]))
            results = xception.decode_predictions(predictions,top=1)
            
            # build result
            data["predictions"] = []
            for (_, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0')