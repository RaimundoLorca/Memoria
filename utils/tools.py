# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 19:41:27 2022

@author: raimu
"""

import numpy as np


def train_test_split(X, y, label_i, label_f, test_i, r):
    
    """
    
    Define una combinación de conjunto de entrenamiento y testeo al definir una
    ventana de tiempo dentro para el conjunto de testeo.
    
    :param np.array X:
        base de datos normalizada.
    :param np.array y:
        etiquetas de los datos.
    :param int label_i:
        número de registro de inucio de etiquetas con fallas.
    :param int label_f:
        número de registro final de etiquetas con fallas.
    :param int test_i:
        número de registro de incio de ventana de tiempo.
    :param float r:
        porcentaje de datos que son parte del conjunto de testeo.
            
    """
    
    test_f=test_i+round((label_f-label_i)*r)
    
    #Obtener datos de testeo que no pertenezcan a la ventana
    labels_test=[]
    while len(labels_test)<(round(len(X)*r)-(test_f-test_i)):
        if np.random.random()<(np.mean([label_i,label_f])/len(X)):
            b=np.random.randint(0,label_i)
            if not b in labels_test:
                labels_test.append(b)
        else:
            c=np.random.randint(label_f+1,len(X))
            if not c in labels_test:
                labels_test.append(c)          
                
    #Unir labels testeo
    labels_v=list(range(test_i,test_f))
    labels_test=list(np.concatenate((labels_test,labels_v)))
    labels_test.sort()
                
    #Definir labels entrenamiento
    labels_train=list(range(len(X)))
    for i in range(len(labels_test)):
        labels_train.remove(labels_test[i])

    X_train = X[labels_train]
    y_train = y[labels_train]

    X_test = X[labels_test]
    y_test = y[labels_test]
    
    return X_train, y_train, X_test, y_test

