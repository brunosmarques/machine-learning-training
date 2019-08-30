# General tools imports
from __future__ import division, print_function
import os
import glob
import re
import numpy as np

# Flask import
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Keras models and tools import
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Tensorflow import
import tensorflow 

# Tensorflow workaround (needed for Win10)
global graph,model
graph = tensorflow.get_default_graph()

# Flash configuration
app = Flask(__name__)

# Model ResNet50 (generic purpose image model already trained)
MODEL_PATH = 'models/your_model.h5'
model = ResNet50(weights='imagenet')

def model_predict(img_path, model):
    """Predict category of an image based on a image path and a model"""

    # image normalization/manipulation
    img = image.load_img(img_path, target_size=(224, 224))    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')

    # do the prediction
    with graph.as_default():
        preds = model.predict(x)        

    # returns the prediction
    return preds

# Default Flask route, index.html
@app.route('/')
def index():
    return render_template('index.html')

# Predict Flask route, actioned by button in index
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        print(preds)

        pred_class = decode_predictions(preds, top=1)        
        result = str(pred_class[0][0][1])
        return result
    return None