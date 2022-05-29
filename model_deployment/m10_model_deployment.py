#!/usr/bin/python

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pandas as pd
import joblib
import sys
import os

def predict_proba(Year,Mileage,State,Make,Model):

    reg = joblib.load(os.path.dirname(__file__) + '/carprice_reg.pkl') 
    d = {'Year': [Year], 'Mileage': [Mileage],'State': [State], 'Make': [Make], 'Model': [Model]}
    df = pd.DataFrame(data=d)
    
  
    # Create features
    le_State = LabelEncoder()
    le_Make = LabelEncoder()
    le_Model = LabelEncoder()
    df['State'] = le_State.fit_transform(df.State)
    df['Make'] = le_Make.fit_transform(df.Make)
    df['Model'] = le_Model.fit_transform(df.Model)
    df_2=df[['Year','Mileage','State','Make','Model']]

    # Make prediction
    p1 = reg.predict(df_2)

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 0:
        print('Please add Year,Mileage,State,Make,Model')
        
    else:

        df = sys.argv[5]

        p1 = predict_proba(Year,Mileage,State,Make,Model)
        
        print(df)
        print('Car Price: ', p1)
