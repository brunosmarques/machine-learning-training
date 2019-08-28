from __future__ import division, print_function
import os
import glob
import re
import numpy as np

from flask import Flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

import tensorflow 
global graph,model
graph = tensorflow.get_default_graph()



app = Flask(__name__)
MODEL_PATH = 'models/your_model.h5'
model = ResNet50(weights='imagenet')

def model_predict(img_path, model):
        #img_path = "C:\\Users\\Bruno\\Desktop\\car.jpg"
        print(img_path)
        img = image.load_img(img_path, target_size=(224, 224))
        
        x = image.img_to_array(img)
        #x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='caffe')

        with graph.as_default():
            preds = model.predict(x)        

        return preds

@app.route('/')
def index():
    return render_template('index.html')

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