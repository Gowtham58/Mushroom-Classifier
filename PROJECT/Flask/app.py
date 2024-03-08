import numpy as np
import os
import sys
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from PIL import *


model = load_model('.\Flask\mushroom.h5')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('predict.html')

@app.route('/predict', methods = ["GET", "POST"])
def res():
    if request.method=="POST":
        f = request.files['images']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        filepath2 = os.path.join(basepath, 'static','uploads', f.filename)
        f.save(filepath)
        
        
        img1 = image.load_img(filepath)
        img = image.load_img(filepath, target_size = (224, 224, 3))
        img1.save(filepath2)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)

        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis = 1)
        
        index = ['Boletus', 'Lactarius', 'Russula']
        
        result = str(index[prediction[0]])
        print(result)

        return render_template('output.html', prediction = result, fname = f.filename)
    
if __name__=="__main__":
    app.run(debug=True)