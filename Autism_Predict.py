import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_models():
    folder_path = os.getcwd() + "/Models"        
    autism_model = pickle.load(open(folder_path+"/Autisum_Model.sav", 'rb'))
    return autism_model

def predict_data(person_info):
    autism_model = load_models()
    label_encoder = LabelEncoder()
    person = pd.DataFrame([person_info])
    person["gender"] = label_encoder.fit_transform(person['gender'])
    person["ethnicity"] = label_encoder.fit_transform(person['ethnicity'])
    person["jundice"] = label_encoder.fit_transform(person['jundice'])

    autism_predict = autism_model.predict([[person.iloc[0]['gender'], person.iloc[0]['ethnicity'], person.iloc[0]['jundice'], person.iloc[0]['age']]])
    
    dataDict = {"autism": autism_predict[0]}
    return dataDict
