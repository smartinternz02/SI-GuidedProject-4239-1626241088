# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:56:08 2020

@author: Adminr
"""
# importing the necessary dependencies
import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__) # initializing a flask app
model = pickle.load(open('cement.pkl', 'rb'))

@app.route('/')# route to display the home page
def home():
    return render_template('home.html') #rendering the home page
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('index1.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')
@app.route('/predict',methods=['POST']) # route to show the predictions in a web UI

def index():
            #  reading the inputs given by the user
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                     'Superplasticizer','Coarse Aggregate', 'Fine Aggregate','Age']
    x = pd.DataFrame(features_value, columns=features_name)
    x_log=np.log(x)#performing log transformation
    # predictions using the loaded model file
    prediction=model.predict(x_log)
    print('prediction is', prediction)
    # showing the prediction results in a UI
    return render_template('result2.html',prediction_text=prediction)

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    #app.run(debug=False) # running the app
    app.run('0.0.0.0',8080) #local host 8080