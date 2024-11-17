from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, SGDRegressor

app = Flask(__name__)

# Load pre-trained models and transformations
ridge_model = pickle.load(open('models/ridge_model.pkl', 'rb'))
lasso_model = pickle.load(open('models/lasso_model.pkl', 'rb'))
sgd_model = pickle.load(open('models/sgd_model.pkl', 'rb'))
poly = pickle.load(open('models/poly.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Define a list of models
models = {
    'ridge': ridge_model,
    'lasso': lasso_model,
    'sgd': sgd_model
}

# Define the categorical and numerical feature lists
categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']

# Define dropdown options for categorical features
dropdown_options = {
    'Item_Fat_Content': ['Low Fat', 'Regular'],
    'Item_Type': ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 'Snack Foods', 
                  'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 
                  'Others', 'Seafood'],
    'Outlet_Size': ['Small', 'Medium', 'High'],
    'Outlet_Location_Type': ['Tier 1', 'Tier 2', 'Tier 3'],
    'Outlet_Type': ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store']
}

@app.route('/')
def index():
    return render_template('index.html', dropdown_options=dropdown_options, models=models.keys(), prediction=None)
    
@app.template_filter('currency')
def currency_format(value):
    return "${:,.2f}".format(value)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected model
    model_choice = request.form.get('model')
    model = models[model_choice]
    
    # Gather and prepare user input
    data = {}
    for feature in numerical_features:
        data[feature] = float(request.form.get(feature))
    for feature in categorical_features:
        data[feature] = request.form.get(feature)
    
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([data])

    # Apply polynomial transformation on numerical features only
    numerical_data = input_df[numerical_features]
    poly_features = poly.transform(numerical_data)
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_features))

    # Add back the categorical features to poly_df
    for feature in categorical_features:
        poly_df[feature] = input_df[feature]

    # Apply one-hot encoding for categorical features
    poly_df = pd.get_dummies(poly_df)
    
    # Align poly_df with the expected columns of the scaler
    expected_columns = scaler.feature_names_in_
    for col in expected_columns:
        if col not in poly_df.columns:
            poly_df[col] = 0  # Add missing columns with a default value of 0
    
    # Ensure columns are in the correct order
    poly_df = poly_df[expected_columns]

    # Apply scaling and make prediction
    scaled_features = scaler.transform(poly_df)
    prediction = model.predict(scaled_features)[0]

    return render_template('index.html', prediction=prediction, dropdown_options=dropdown_options, models=models.keys())

if __name__ == '__main__':
    app.run(debug=True)
