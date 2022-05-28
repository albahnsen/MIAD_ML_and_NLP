#!/usr/bin/python

import numpy as np
import pandas as pd
import joblib
import sys
import os
import nltk
from nltk.corpus import stopwords
import re
import random
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    # Remueve un valor en particular encontrado en la exploracion " N"    
    text = re.sub("[.N]"," ",text)

    # Remueve todo menos el alfabeto
    text = re.sub("[^a-zA-Z]"," ",text) 

    # Remueve los espacios en blanco
    text = ' '.join(text.split()) 

    # Remueve backslash-apostrofe
    text = re.sub("\'", "", text) 

    # convierte el texto en minuscula
    text = text.lower() 

    return text    

def remove_stopwords(text):
    text = text
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def extraer_prediccion(proba, clases):
    datos_test = proba[0].shape[0]
    resultado = np.full([datos_test, clases], None)
    for j in range(datos_test):
        prediccion_i = [] 
        for i in range(clases):
            prediccion_i.append(proba[i][j][1])
        resultado[j] = prediccion_i
    
    return resultado

def predict_proba(text):
    
    cleanning = joblib.load('../model_deployment/cleanning.pkl') 
    sw = joblib.load('../model_deployment/stop_wors.pkl')
    tf_vect = joblib.load('../model_deployment/vectorizer.pkl')
    clf = joblib.load('../model_deployment/genre_clf.pkl')



    df = pd.DataFrame(data={'plot':0},index=[0])
    df['plot']=text


    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    clases = len(cols)

    text=clean_text(text)
    text=remove_stopwords(text)
    df['plot']=text

    text=tf_vect.transform(df['plot'])
    p1=clf.predict_proba(text)
    Prediccion_API_prob = extraer_prediccion(p1, clases)
    Prediccion_API_prob = Prediccion_API_prob[0]

    # Identificación de géneros

    Prediccion_API = []
    for i in range(len(Prediccion_API_prob)):
        if Prediccion_API_prob[i] >= 0.5:
            Prediccion_API.append(cols[i][2:])

    # Resultado

    return Prediccion_API



if __name__ == "__main__":
    
    if len(sys.argv) == 0:
        print('Please enter the movie plot')
        
    else:

        text = sys.argv[1]

        Prediccion_API = predict_proba(text)
                
        print(text)
        print('this description is categorized into', len(Prediccion_API), 'different genres, which are:', Prediccion_API)
