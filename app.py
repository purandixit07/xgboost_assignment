# importing the necessary dependencies
from flask import Flask, render_template, request, jsonify
import xgboost as xgb
from xgboost import XGBClassifier
from flask_cors import CORS, cross_origin
# import sklearn
# from sklearn.linear_model import LinearRegression
import pickle

# print (sklearn.__version__)


app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
# @cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
# @cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            age = float(request.form['age'])
            workclass = float(request.form['workclass'])
            fnlwgt = float(request.form['fnlwgt'])
            education = float(request.form['education'])
            education_num = float(request.form['education_num'])
            marital_status = float(request.form['marital_status'])
            occupation = float(request.form['occupation'])
            relationship = float(request.form['relationship'])
            race = float(request.form['race'])
            sex = float(request.form['sex'])
            capital_gain = float(request.form['capital_gain'])
            capital_loss = float(request.form['capital_loss'])
            hours_per_week = float(request.form['hours_per_week'])
            native_country = float(request.form['native_country'])

            filename = 'xgboost_model.pickle'
            filename_scaler = 'xgb_scaler_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            scaler_model = pickle.load(open(filename_scaler, 'rb'))
            # predictions using the loaded model file
            a = scaler_model.transform([[age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country]])
            prediction = loaded_model.predict(a)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html', prediction=prediction[0])
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


# if __name__ == "__main__":
#     #app.run(host='127.0.0.1', port=8001, debug=True)
# 	app.run(debug=True) # running the app
if __name__ == '__main__':
    app.run(debug=True)
