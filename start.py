from algorithms.classification.auto_sklearn import autoSklearn_class
from algorithms.classification.auto_keras import autokeras_class
#from algorithms.classification.h20 import h20_class
#from algorithms.classification.mlbox import mlbox_class
from algorithms.classification.tpot import tpot_class

import openml
import os.path
from os import path
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.datasets import fetch_openml
from os import listdir
from os.path import isfile, join

openml_list = openml.datasets.list_datasets()  # returns a dict

datalist = pd.DataFrame.from_dict(openml_list, orient="index")
datalist = datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]
df_id_name = datalist[datalist.NumberOfInstances > 4000].sort_values(["NumberOfInstances"]).head(n=1)

df_good = 0
df_bad = 0

#scarico i dataset 
for index, row in df_id_name.iterrows():
    df_id = row['did']
    df_name = row['name']
    df_good = 0
    df_bad = 0
    print('Dataset ID: ' + str(df_id) + ' name: ' + df_name)
    try:
        X, y = fetch_openml(data_id=df_id, as_frame=True, return_X_y=True)
        y = y.to_frame()
    except:
        print("bad df\n")
        df_bad+=1

    if pd.api.types.infer_dtype(y[y.columns[0]]) != "categorical":
        #regressione
        file_dir =  './datasets/regression/'
    else:
        #classificazione
        file_dir =  './datasets/classification/'
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    fullname = os.path.join(file_dir, row['name'] + '.csv')
    #print(fullname)
    
    print("good df\n")
    df_good+=1

    if not os.path.exists(fullname):
        X[y.columns[0]] = y
        X.to_csv(fullname, index=False, header=True)
        
    
print('Goode df: ' + str(df_good) + '    baad df: ' + str(df_bad) + '\n')


datasets = [f for f in listdir('./datasets/classification/') if isfile(join('./datasets/classification/', f))]
for dt in datasets:
    df = pd.read_csv(os.path.join('./datasets/classification/', dt))
    #print(autoSklearn_class(df)) # ritorna l'accuracy
    print(autokeras_class(df)) # ritorna l'accuracy
    #print(h20_class(df)) 
    #rint(mlbox_class(df)) 
    print(tpot_class(df)) # ritorna l'accuracy


#datasets = [f for f in listdir('../dataset/regression/') if isfile(join('../dataset/clasification/', f))]
#for dt in datasets:
#    df = pd.read_csv(dt)
#    print(autoSklearn(df))
#    print(autokeras(df))
#    print(h20(df))
#    print(mlbox(df))
#    print(tpot(df))


#stampo i grafici per confrontare i risultati