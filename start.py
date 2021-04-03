from algorithms.classification.auto_sklearn import autoSklearn_class
from algorithms.classification.auto_keras import autokeras_class
from algorithms.classification.h20 import h20_class
from algorithms.classification.mlbox import mlbox_class
from algorithms.classification.tpot import tpot_class

import openml
import os.path
from os import path
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from os import listdir
from os.path import isfile, join

openml_list = openml.datasets.list_datasets()  # returns a dict

datalist = pd.DataFrame.from_dict(openml_list, orient="index")
datalist = datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

df_id_name = datalist[datalist.NumberOfInstances > 4000].sort_values(["NumberOfInstances"]).head(n=50)[:,1:2]

#scarico i dataset 
for index, row in df_id_name.iterrows():    
    X, y = fetch_openml(data_id=row['did'], as_frame=True, return_X_y=True)
    y = y.to_frame()

    if pd.api.types.infer_dtype(y[y.columns[0]]) != "categorical":
        #regressione
        file_path =  '../dataset/regression/' + row['name'] + '.csv'
    else:
        #classificazione
        file_path =  '../dataset/clasification/' + row['name'] + '.csv'
        
    if (os.path.exists(file_path) == False):
        X[y.columns[0]] = y
        X.to_csv(file_path, index=False, header=True)


datasets = [f for f in listdir('../dataset/clasification/') if isfile(join('../dataset/clasification/', f))]
for dt in datasets:
    df = pd.read_csv(dt)
    print(autoSklearn_class(df)) # ritorna l'accuracy
    print(autokeras_class(df)) # ritorna l'accuracy
    print(h20_class(df)) 
    print(mlbox_class(df)) 
    print(tpot_class(df)) # ritorna l'accuracy


datasets = [f for f in listdir('../dataset/regression/') if isfile(join('../dataset/clasification/', f))]
for dt in datasets:
    df = pd.read_csv(dt)
    print(autoSklearn(df))
    print(autokeras(df))
    print(h20(df))
    print(mlbox(df))
    print(tpot(df))


#stampo i grafici per confrontare i risultati