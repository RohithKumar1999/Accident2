import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from flask import Flask, jsonify, request
from flask import Flask, render_template
import pickle
import joblib 
dataset=pd.read_excel(r'C:\Users\user\Accident_data\Acc_data_version2 - Copy.xlsx')
dataset=dataset.drop(["State","Day","Time","Weather","Age","OVERLOAD","Vehicle_Age","COMMODITY","Cost % Of Commodity Damage","Packed_in","VEHICLE_TYPE","WEIGHT_CAPACITY(METRIC_TON)","COMMODITY WEIGHT","VEHICLE_HEALTH"],axis=1)
dataset['OVERLOAD SCALE'] = dataset['OVERLOAD SCALE'].astype(float)
dataset['VEHICLE HEALTH SCALE'] = dataset['VEHICLE HEALTH SCALE'].astype(float)
X=pd.DataFrame(dataset.iloc[:,:-1])
y=pd.DataFrame(dataset.iloc[:,-1])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(min_samples_split=4,max_depth=25)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Save the model as a pickle in a file 
joblib.dump(regressor, 'Accident.pkl') 
  
# Load the model from the file 
knn_from_joblib = joblib.load('Accident.pkl')  
  
# Use the loaded model to make predictions 
knn_from_joblib.predict(X_test) 
clf = joblib.load('Accident.pkl')
# app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = clf.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Risk will be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = clf.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
