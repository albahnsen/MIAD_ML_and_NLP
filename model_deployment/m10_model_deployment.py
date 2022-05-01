#!/usr/bin/python
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import sys
import os

def predict_proba(Year,Mileage,State,Make,Model):

    reg = joblib.load(os.path.dirname(__file__) + '/carprice_reg.pkl') 
    d = {'Year': [Year], 'Mileage': [Mileage],'State': [State], 'Make': [Make], 'Model': [Model]}
    df = pd.DataFrame(data=d)
    
  
    # Create features
    # Codificación variables categóricas
    le_State = LabelEncoder()
    le_Make = LabelEncoder()
    le_Model = LabelEncoder()
    df['State_encoded'] = le_State.fit_transform(df.State)
    df['Make_encoded'] = le_Make.fit_transform(df.Make)
    df['Model_encoded'] = le_Model.fit_transform(df.Model)
    df_2=df[['Year',	'Mileage',	'State_encoded','Make_encoded',	'Model_encoded']]

    # Make prediction
    p1 = reg.predict(df_2)

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add Year,Mileage,State,Make,Model')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(df_2)
        
        print(df)
        print('Probability of Phishing: ', p1)
