from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import streamlit as st
import pickle 

import pandas as pd
import numpy as np


df = pd.read_csv('water_potability3.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, 9].values


X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y)

kn_classifier = KNeighborsClassifier()
kn_classifier.fit(X_train5, y_train5)

pickle.dump(kn_classifier,open('kn_classifier.pkl','wb'))