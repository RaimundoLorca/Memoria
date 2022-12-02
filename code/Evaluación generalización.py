# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:00:19 2022

@author: raimu
"""

from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.metrics import classification_report_imbalanced
from imblearn.combine import SMOTEENN, SMOTETomek 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from sklearn.metrics import f1_score, precision_score

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier

# import xgboost as xgb
import joblib 

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import utils 

#Cargar datos
df=pd.read_csv('C:/Users/raimu/Desktop/Memoria/data/Bba118_processed.csv') 

#%% Filtro alta correlación

#Por corriente
a=df['Corriente L1']!=0
df=df[a]

df_original = df 

# Columnas a utilizar
x_axis_labels=['Corriente L1', 'Corriente L2', 'Corriente L3',
       'Flujo Descarga', 'T° Desc Bba. LL', 'T° Desc Bba. LA',
       'T° Desc Mot. LL', 'T° Desc Mot. LL.1', 'Prox. Bba. LLX',
       'Prox. Bba. LLY', 'Prox. Bba. LAX', 'Prox. Bba. LAY', 'Prox. Mot. LAX',
       'Prox. Mot. LAY', 'Prox. Mot LLX', 'Prox. Mot. LLY']

df = df[x_axis_labels]

#Etiquetas
label_i=10925 #10900 considera todo el dia anterior
label_f=11164
 
y=np.zeros((len(df),1))
y[label_i:label_f]=1 #10932:11203 desde el dia antes a las 13:00 aprox

# Transformar a numpy array
df_pd = df
df = df.to_numpy()
 
# Estandarización datos
scaler = StandardScaler().fit(df)
X=scaler.transform(df)


#%% Cargar modelo.
model1 = joblib.load('C:/Users/raimu/Desktop/Memoria/data/Best_Modelo_O1.pkl') # Carga del modelo.

Yp=model1.predict(X)

# Plotear matriz de confusión (Falla=1 y Saludable=0)
cm = confusion_matrix(y, Yp,labels=model1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model1.classes_)
disp.plot()
plt.title('Matriz de confusión OverSampling RandomOverSampler')
plt.show()

print("Modelo oversample RandomOverSampler:")
print(f1_score(y, Yp))
print(sensitivity_score(y, Yp))
print(specificity_score(y, Yp))
print(geometric_mean_score(y.ravel(), Yp))
print(precision_score(y, Yp))
# print('\n')