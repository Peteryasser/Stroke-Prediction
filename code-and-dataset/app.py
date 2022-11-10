import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import RobustScaler

app= Flask(__name__)

model=pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    scaler = RobustScaler()
    float_feathers = [float(x) for x in request.form.values()]
    features = np.array(float_feathers).reshape(-1, 1)
    features = scaler.fit_transform(X=features)
    features = features
    prediction = model.predict(features.T)
    result=""
    if prediction==1:
        result='Sorry but you may have Stroke brain with probability 95%'
    else:
        result='We have a good news, you may don\'t have stroke brain with probability 95%'


    return render_template("index.html" , prediction_text = result)

if __name__ == '__main__':
    app.run()