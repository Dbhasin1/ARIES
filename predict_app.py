import base64
import numpy as np
import tensorflow as tf
import io
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Flatten,Dense, Dropout 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

def get_model():
    global model
    model = load_model('model.h5')
    model._make_predict_function()
    print(" * Model loaded!")
    
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    return image

print(" * Loading Keras model...")
#get_model()
global graph
#graph = tf.get_default_graph()
graph = tf.Graph()
with graph.as_default():
    get_model()
#tf.compat.v1.global_variables_initializer
#tf.initialize_all_variables().run()
#tf.global_variables_initializer().run()
tf.compat.v1.global_variables_initializer()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    preprocessed_image = preprocess_image(image, target_size=(256, 256))
    
    with graph.as_default():
        prediction = model.predict(preprocessed_image).tolist()
    
    response = {
        'prediction': {
            'hundred' : prediction[0],
            'fivehundred' : prediction[1]
        }
    }
    return jsonify(response)

#if __name__ == "__main__":
#    app.run(debug=True)