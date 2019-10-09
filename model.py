# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
dataset=pd.read_excel(r'C:\Users\user\Accident_data\Acc_data_version2 - Copy.xlsx')
dataset=dataset.drop(["State","Day","Time","Weather","Age","Vehicle_Age","COMMODITY","Cost % Of Commodity Damage","Packed_in","VEHICLE_TYPE","WEIGHT_CAPACITY(METRIC_TON)","COMMODITY WEIGHT","VEHICLE_HEALTH"],axis=1)
X=pd.DataFrame(dataset.iloc[:,:-1])
y=pd.DataFrame(dataset.iloc[:,-1])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(min_samples_split=4,max_depth=25)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(regressor, 'Accident.pkl') 
  
# Load the model from the file 
model= joblib.load('Accident.pkl')  
X_test=X.iloc[1336:]
# Use the loaded model to make predictions 
model.predict(X_test) 
