from flask import Flask, request, render_template, url_for
import requests
import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import r2_score

import pickle

# Reference dictionaries
fuel = pickle.load(open('fuel.pkl', 'rb'))
company = pickle.load(open('company.pkl', 'rb'))
year = pickle.load(open('year.pkl', 'rb'))
model = pickle.load(open('name.pkl', 'rb'))
#csv
car=pd.read_csv('cleaned.csv')
# Regressor
reg = pickle.load(open('linreg1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    yr = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=yr, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        brand = request.form.get('company')
        name_ = request.form.get('car_models')
        yop = int(request.form.get('year'))
        dist = int(request.form.get('kilo_driven'))
        ful = request.form.get('fuel_type')
        # Encoding brand,model name and fuel
        brand_enc = company[brand]
        n = model[name_]
        fuel_enc = fuel[ful]
        inputt = [n, brand_enc, yop, dist, fuel_enc]
        # Prediction results
        price = (math.ceil(reg.predict(np.array(inputt).reshape(1, 5))))
        price = price * 25
        return render_template('index.html', price=price, check=1)
    else:
        return render_template('index.html', check=0)


if __name__=="__main__":
    app.run(debug=True)
