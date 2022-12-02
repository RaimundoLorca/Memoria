# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:38:17 2022

@author: raimu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import SMOTEENN, SMOTETomek 
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

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

#%% Filtro alta correlación y definir etiquetas

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

#%% Definir conjuntos entrenamiento y testeo segun ventana de tiempo

# Transformar a numpy array
df_pd = df
df = df.to_numpy()
 
# Estandarización datos
scaler = StandardScaler().fit(df)
X=scaler.transform(df)

# Arrays para guardar resultados
Resultados=np.zeros(shape=(5*22,10))


for i in range(10):
    
    #Obtener datos de testeo que no pertenezcan a la ventana
    X_train, y_train, X_test, y_test = utils.train_test_split(X, y, label_i=10925, 
                                                        label_f=11164, test_i=10945, r=0.25)
    
    #------------------------------------------------------------------------------------
    # Caso base
    model1 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)
    
    model1 = model1.fit(X_train, y_train.ravel())
    
    Yp=model1.predict(X_test)
    
    Resultados[0,i]=f1_score(y_test, Yp)
    Resultados[1,i]=sensitivity_score(y_test, Yp)
    Resultados[2,i]=specificity_score(y_test, Yp)
    Resultados[3,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[4,i]=precision_score(y_test, Yp)
    
    if Resultados[0,i] == max(Resultados[0]):
        #Guardar modelo
        joblib.dump(model1, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_CB.pkl')
    
    #------------------------------------------------------------------------------------
    # Caso Oversample 1
    oversample = RandomOverSampler()
    
    #Aplicar metodo de oversampling
    X_train_o1, y_train_o1 = oversample.fit_resample(X_train, y_train)
    
    model2 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)
    
    model2 = model2.fit(X_train_o1, y_train_o1.ravel())
    
    Yp=model2.predict(X_test)
    
    Resultados[5,i]=f1_score(y_test, Yp)
    Resultados[6,i]=sensitivity_score(y_test, Yp)
    Resultados[7,i]=specificity_score(y_test, Yp)
    Resultados[8,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[9,i]=precision_score(y_test, Yp)

    if Resultados[5,i] == max(Resultados[5]):
        #Guardar modelo
        joblib.dump(model2, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_OverSampler_RandomOverSampler.pkl') 
        
    #------------------------------------------------------------------------------------
    # Caso Oversample 2
    oversample = SMOTE()
    
    #Aplicar metodo de oversampling
    X_train_o2, y_train_o2 = oversample.fit_resample(X_train, y_train)

    model3 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model3 = model3.fit(X_train_o2, y_train_o2.ravel())

    Yp=model3.predict(X_test)
    
    Resultados[10,i]=f1_score(y_test, Yp)
    Resultados[11,i]=sensitivity_score(y_test, Yp)
    Resultados[12,i]=specificity_score(y_test, Yp)
    Resultados[13,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[14,i]=precision_score(y_test, Yp)

    if Resultados[10,i] == max(Resultados[10]):
        #Guardar modelo
        joblib.dump(model3, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_OverSampler_SMOTE.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Oversample 3
    oversample = ADASYN()
    
    #Aplicar metodo de oversampling
    X_train_o3, y_train_o3 = oversample.fit_resample(X_train, y_train)

    model4 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model4 = model4.fit(X_train_o3, y_train_o3.ravel())

    Yp=model4.predict(X_test)
    
    Resultados[15,i]=f1_score(y_test, Yp)
    Resultados[16,i]=sensitivity_score(y_test, Yp)
    Resultados[17,i]=specificity_score(y_test, Yp)
    Resultados[18,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[19,i]=precision_score(y_test, Yp)

    if Resultados[15,i] == max(Resultados[15]):
        #Guardar modelo
        joblib.dump(model4, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_OverSampler_ADASYN.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Oversample 4
    oversample = BorderlineSMOTE()
    
    #Aplicar metodo de oversampling
    X_train_o4, y_train_o4 = oversample.fit_resample(X_train, y_train)

    model5 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model5 = model5.fit(X_train_o4, y_train_o4.ravel())

    Yp=model5.predict(X_test)
    
    Resultados[20,i]=f1_score(y_test, Yp)
    Resultados[21,i]=sensitivity_score(y_test, Yp)
    Resultados[22,i]=specificity_score(y_test, Yp)
    Resultados[23,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[24,i]=precision_score(y_test, Yp)

    if Resultados[20,i] == max(Resultados[20]):
        #Guardar modelo
        joblib.dump(model5, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_OverSampler_BorderlineSMOTE.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Oversample 5
    oversample = SVMSMOTE()
    
    #Aplicar metodo de oversampling
    X_train_o5, y_train_o5 = oversample.fit_resample(X_train, y_train)

    model6 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model6 = model6.fit(X_train_o5, y_train_o5.ravel())

    Yp=model6.predict(X_test)
    
    Resultados[25,i]=f1_score(y_test, Yp)
    Resultados[26,i]=sensitivity_score(y_test, Yp)
    Resultados[27,i]=specificity_score(y_test, Yp)
    Resultados[28,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[29,i]=precision_score(y_test, Yp)

    if Resultados[25,i] == max(Resultados[25]):
        #Guardar modelo
        joblib.dump(model6, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_OverSampler_SVMSMOTE.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 1
    undersample = ClusterCentroids()
    
    #Aplicar metodo de undersampling
    X_train_u1, y_train_u1 = undersample.fit_resample(X_train, y_train)

    model7 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model7 = model7.fit(X_train_u1, y_train_u1.ravel())

    Yp=model7.predict(X_test)
    
    Resultados[30,i]=f1_score(y_test, Yp)
    Resultados[31,i]=sensitivity_score(y_test, Yp)
    Resultados[32,i]=specificity_score(y_test, Yp)
    Resultados[33,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[34,i]=precision_score(y_test, Yp)

    if Resultados[30,i] == max(Resultados[30]):
        #Guardar modelo
        joblib.dump(model7, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_ClusterCentroids.pkl') 

    
    #------------------------------------------------------------------------------------
    # Caso Undersample 2
    undersample = CondensedNearestNeighbour()
    
    #Aplicar metodo de undersampling
    X_train_u2, y_train_u2 = undersample.fit_resample(X_train, y_train)

    model8 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model8 = model8.fit(X_train_u2, y_train_u2.ravel())

    Yp=model8.predict(X_test)
    
    Resultados[35,i]=f1_score(y_test, Yp)
    Resultados[36,i]=sensitivity_score(y_test, Yp)
    Resultados[37,i]=specificity_score(y_test, Yp)
    Resultados[38,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[39,i]=precision_score(y_test, Yp)

    if Resultados[35,i] == max(Resultados[35]):
        #Guardar modelo
        joblib.dump(model8, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_CondensedNearestNeighbour.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 3
    undersample = EditedNearestNeighbours()
    
    #Aplicar metodo de undersampling
    X_train_u3, y_train_u3 = undersample.fit_resample(X_train, y_train)

    model9 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model9 = model9.fit(X_train_u3, y_train_u3.ravel())

    Yp=model9.predict(X_test)
    
    Resultados[40,i]=f1_score(y_test, Yp)
    Resultados[41,i]=sensitivity_score(y_test, Yp)
    Resultados[42,i]=specificity_score(y_test, Yp)
    Resultados[43,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[44,i]=precision_score(y_test, Yp)

    if Resultados[40,i] == max(Resultados[40]):
        #Guardar modelo
        joblib.dump(model9, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_EditedNearestNeighbours.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 4
    undersample = RepeatedEditedNearestNeighbours()
    
    #Aplicar metodo de undersampling
    X_train_u4, y_train_u4 = undersample.fit_resample(X_train, y_train)

    model10 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model10 = model10.fit(X_train_u4, y_train_u4.ravel())

    Yp=model10.predict(X_test)
    
    Resultados[45,i]=f1_score(y_test, Yp)
    Resultados[46,i]=sensitivity_score(y_test, Yp)
    Resultados[47,i]=specificity_score(y_test, Yp)
    Resultados[48,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[49,i]=precision_score(y_test, Yp)

    if Resultados[45,i] == max(Resultados[45]):
        #Guardar modelo
        joblib.dump(model10, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_RepeatedEditedNearestNeighbours.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 5
    undersample = AllKNN()
    
    #Aplicar metodo de undersampling
    X_train_u5, y_train_u5 = undersample.fit_resample(X_train, y_train)

    model11 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model11 = model11.fit(X_train_u5, y_train_u5.ravel())

    Yp=model11.predict(X_test)
    
    Resultados[50,i]=f1_score(y_test, Yp)
    Resultados[51,i]=sensitivity_score(y_test, Yp)
    Resultados[52,i]=specificity_score(y_test, Yp)
    Resultados[53,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[54,i]=precision_score(y_test, Yp)

    if Resultados[50,i] == max(Resultados[50]):
        #Guardar modelo
        joblib.dump(model11, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_AllKNN.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 6
    undersample = NearMiss()
    
    #Aplicar metodo de undersampling
    X_train_u6, y_train_u6 = undersample.fit_resample(X_train, y_train)

    model12 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model12 = model12.fit(X_train_u6, y_train_u6.ravel())

    Yp=model12.predict(X_test)
    
    Resultados[55,i]=f1_score(y_test, Yp)
    Resultados[56,i]=sensitivity_score(y_test, Yp)
    Resultados[57,i]=specificity_score(y_test, Yp)
    Resultados[58,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[59,i]=precision_score(y_test, Yp)

    if Resultados[55,i] == max(Resultados[55]):
        #Guardar modelo
        joblib.dump(model12, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_NearMiss.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 7
    undersample = NeighbourhoodCleaningRule()
    
    #Aplicar metodo de undersampling
    X_train_u7, y_train_u7 = undersample.fit_resample(X_train, y_train)

    model13 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model13 = model13.fit(X_train_u7, y_train_u7.ravel())

    Yp=model13.predict(X_test)
    
    Resultados[60,i]=f1_score(y_test, Yp)
    Resultados[61,i]=sensitivity_score(y_test, Yp)
    Resultados[62,i]=specificity_score(y_test, Yp)
    Resultados[63,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[64,i]=precision_score(y_test, Yp)

    if Resultados[60,i] == max(Resultados[60]):
        #Guardar modelo
        joblib.dump(model13, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_NeighbourhoodCleaningRule.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 8
    undersample = OneSidedSelection()
    
    #Aplicar metodo de undersampling
    X_train_u8, y_train_u8 = undersample.fit_resample(X_train, y_train)

    model14 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model14 = model14.fit(X_train_u8, y_train_u8.ravel())

    Yp=model14.predict(X_test)
    
    Resultados[65,i]=f1_score(y_test, Yp)
    Resultados[66,i]=sensitivity_score(y_test, Yp)
    Resultados[67,i]=specificity_score(y_test, Yp)
    Resultados[68,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[69,i]=precision_score(y_test, Yp)

    if Resultados[65,i] == max(Resultados[65]):
        #Guardar modelo
        joblib.dump(model14, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_OneSidedSelection.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 9
    undersample = RandomUnderSampler()
    
    #Aplicar metodo de undersampling
    X_train_u9, y_train_u9 = undersample.fit_resample(X_train, y_train)

    model15 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model15 = model15.fit(X_train_u9, y_train_u9.ravel())

    Yp=model15.predict(X_test)
    
    Resultados[70,i]=f1_score(y_test, Yp)
    Resultados[71,i]=sensitivity_score(y_test, Yp)
    Resultados[72,i]=specificity_score(y_test, Yp)
    Resultados[73,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[74,i]=precision_score(y_test, Yp)

    if Resultados[70,i] == max(Resultados[70]):
        #Guardar modelo
        joblib.dump(model15, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_RandomUnderSampler.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Undersample 10
    undersample = TomekLinks()
    
    #Aplicar metodo de undersampling
    X_train_u10, y_train_u10 = undersample.fit_resample(X_train, y_train)

    model16 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model16 = model16.fit(X_train_u10, y_train_u10.ravel())

    Yp=model16.predict(X_test)
    
    Resultados[75,i]=f1_score(y_test, Yp)
    Resultados[76,i]=sensitivity_score(y_test, Yp)
    Resultados[77,i]=specificity_score(y_test, Yp)
    Resultados[78,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[79,i]=precision_score(y_test, Yp)

    if Resultados[75,i] == max(Resultados[75]):
        #Guardar modelo
        joblib.dump(model16, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_UnderSampler_TomekLinks.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Over-Under-Sampling 1
    overundersample = SMOTEENN()
    
    #Aplicar metodo de over-undersampling
    X_train_ou1, y_train_ou1 = overundersample.fit_resample(X_train, y_train)

    model17 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model17 = model17.fit(X_train_ou1, y_train_ou1.ravel())

    Yp=model17.predict(X_test)
    
    Resultados[80,i]=f1_score(y_test, Yp)
    Resultados[81,i]=sensitivity_score(y_test, Yp)
    Resultados[82,i]=specificity_score(y_test, Yp)
    Resultados[83,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[84,i]=precision_score(y_test, Yp)

    if Resultados[80,i] == max(Resultados[80]):
        #Guardar modelo
        joblib.dump(model17, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_OverUnderSampler_SMOTEENN.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Over-Under-Sampling 2
    overundersample = SMOTETomek()
    
    #Aplicar metodo de over-undersampling
    X_train_ou2, y_train_ou2 = overundersample.fit_resample(X_train, y_train)

    model18 = MLPClassifier(learning_rate='constant',early_stopping=True,
                          n_iter_no_change=50,max_iter=1000)

    model18 = model18.fit(X_train_ou2, y_train_ou2.ravel())

    Yp=model18.predict(X_test)
    
    Resultados[85,i]=f1_score(y_test, Yp)
    Resultados[86,i]=sensitivity_score(y_test, Yp)
    Resultados[87,i]=specificity_score(y_test, Yp)
    Resultados[88,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[89,i]=precision_score(y_test, Yp)

    if Resultados[85,i] == max(Resultados[85]):
        #Guardar modelo
        joblib.dump(model18, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_OverUnderSampler_SMOTETomek.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Ensemble Learning 1
    
    #Ensemble balanceado con DecisionTree Classifier
    model19 = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                       sampling_strategy='auto',
                                       replacement=False,
                                       random_state=0)

    model19 = model19.fit(X_train, y_train.ravel())

    Yp=model19.predict(X_test)
    
    Resultados[90,i]=f1_score(y_test, Yp)
    Resultados[91,i]=sensitivity_score(y_test, Yp)
    Resultados[92,i]=specificity_score(y_test, Yp)
    Resultados[93,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[94,i]=precision_score(y_test, Yp)

    if Resultados[90,i] == max(Resultados[90]):
        #Guardar modelo
        joblib.dump(model19, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_EnsembleLearning_BalancedBaggingClassifier.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Ensemble Learning 2
    
    #Ensemble balanceado con RandomForestClassifier
    model20 = BalancedRandomForestClassifier(n_estimators=100, random_state=0)

    model20 = model20.fit(X_train, y_train.ravel())

    Yp=model20.predict(X_test)
    
    Resultados[95,i]=f1_score(y_test, Yp)
    Resultados[96,i]=sensitivity_score(y_test, Yp)
    Resultados[97,i]=specificity_score(y_test, Yp)
    Resultados[98,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[99,i]=precision_score(y_test, Yp)

    if Resultados[95,i] == max(Resultados[95]):
        #Guardar modelo
        joblib.dump(model20, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_EnsembleLearning_BalancedRandomForestClassifier.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Ensemble Learning 3
    
    #Ensemble balanceado con RUSBoostClassifier
    base_estimator = AdaBoostClassifier(n_estimators=10)
    model21 = RUSBoostClassifier(random_state=0, n_estimators=10, base_estimator=base_estimator)

    model21 = model21.fit(X_train, y_train.ravel())

    Yp=model21.predict(X_test)
    
    Resultados[100,i]=f1_score(y_test, Yp)
    Resultados[101,i]=sensitivity_score(y_test, Yp)
    Resultados[102,i]=specificity_score(y_test, Yp)
    Resultados[103,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[104,i]=precision_score(y_test, Yp)

    if Resultados[100,i] == max(Resultados[100]):
        #Guardar modelo
        joblib.dump(model21, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_EnsembleLearning_RUSBoostClassifier.pkl') 

    #------------------------------------------------------------------------------------
    # Caso Ensemble Learning 4
    
    #Ensemble balanceado con RUSBoostClassifier
    base_estimator = AdaBoostClassifier(n_estimators=10)
    model22 = EasyEnsembleClassifier(n_estimators=10, base_estimator=base_estimator)
    
    model22 = model22.fit(X_train, y_train.ravel())

    Yp=model22.predict(X_test)
    
    Resultados[105,i]=f1_score(y_test, Yp)
    Resultados[106,i]=sensitivity_score(y_test, Yp)
    Resultados[107,i]=specificity_score(y_test, Yp)
    Resultados[108,i]=geometric_mean_score(y_test.ravel(), Yp)
    Resultados[109,i]=precision_score(y_test, Yp)

    if Resultados[105,i] == max(Resultados[105]):
        #Guardar modelo
        joblib.dump(model22, 'C:\\Users\\raimu\\Desktop\\Memoria\\Data\\Mejor_Modelo_EnsembleLearning_EasyEnsembleClassifier.pkl') 
