import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os 

def austim_Detector_model():
    autism = pd.read_csv("csv_result-Autism-Adult-Data.csv", encoding='ISO-8859-1', index_col="id")
    label_encoder = preprocessing.LabelEncoder()
    autism = autism.replace('?', None)
    autism.dropna(inplace=True)
    autism["gender"] = label_encoder.fit_transform(autism['gender'])
    autism["ethnicity"] = label_encoder.fit_transform(autism['ethnicity'])
    autism["jundice"] = label_encoder.fit_transform(autism['jundice'])
    feature_names = ['gender', 'ethnicity', 'jundice', 'age']
    target_name = 'austim' 
    X = autism[feature_names]
    y = autism[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=25)
    par = LogisticRegression()
    par.fit(X_train, y_train)
    
    pred = par.predict(X_test)
    print(accuracy_score(pred, y_test))

    filename = os.getcwd() +"/"+"Autisum_Model.sav"
    pickle.dump(par, open(filename, 'wb'))

austim_Detector_model()