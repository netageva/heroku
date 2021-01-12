from flask import Flask
from flask import request
import pandas as pd
import numpy as np
import pickle
import json
import os
import gunicorn

app = Flask(__name__)

X_test = pd.read_csv('X_test.csv')
model_pkl = open('rf_model.sav', 'rb')
clf = pickle.load(model_pkl)

@app.route('/predict_single')
def predict_single():
    feat = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal']

    array_to_pred = np.zeros((len(feat)))

    for f in range(len(feat)):
        array_to_pred[f] = request.args.get(feat[f])

    pred = clf.predict(array_to_pred.reshape(1,-1))
    ans = ''
    if pred == 0:
        ans = 'No heart disease'
    else:
        ans = 'Heart disease'

    return 'Prediction for single sample: {} ({})'.format(pred,ans)

@app.route("/multiple_pred", methods=['GET', "POST"])
def multiple_predictions():
    params_json = request.get_json()
    params = json.loads(params_json)
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal']

    n_rows = len(params['age'])
    n_cols = len(features)
    X_to_pred = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            X_to_pred[i, j] = params[features[j]][i]

    y_pred = clf.predict(X_to_pred)
    y_pred_json = json.dumps(y_pred.tolist())

    return y_pred_json
if __name__ == '__main__':
    port = os.environ.get('PORT')
    app.run(host='0.0.0.0', port=int(port))

#chaecking