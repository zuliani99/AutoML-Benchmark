#pip3 install openml

#from algorithms.classification.auto_sklearn import autoSklearn_class
from algorithms.classification.auto_keras import autokeras_class
#from algorithms.classification.h2o import h2o_class
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


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import autosklearn.classification

if __name__ == '__main__':   

    print("--------------------start------------------")
    
    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > 40000].sort_values(["NumberOfInstances"]).head(5)

    df_good = 0
    df_bad = 0

    list_calss = []
    list_reg = []

    res_class = {}
    res_res = {}

    test = False

    print(df_id_name.info())

    for index, row in df_id_name.iterrows():
        df_id = row['did']
        df_name = row['name']

        print('Dataset ID: ' + str(df_id) + ' name: ' + df_name)
        try:
            X, y = fetch_openml(data_id=df_id, as_frame=True, return_X_y=True)
            y = y.to_frame()

            if pd.api.types.infer_dtype(y[y.columns[0]]) != "categorical":
                file_dir =  './datasets/regression/'
            else:
                file_dir =  './datasets/classification/'
            
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            fullname = os.path.join(file_dir, row['name'] + '.csv')

            print("good df\n")
            df_good+=1

            if not os.path.exists(fullname):
                X[y.columns[0]] = y
                X.to_csv(fullname, index=False, header=True)

            if(file_dir == './datasets/regression/'):
                list_reg.append(fullname)
            else:
                list_calss.append(fullname)

        except:
            print("bad df\n")
            df_bad+=1

    print('Good df: ' + str(df_good) + '    baad df: ' + str(df_bad) + '\n')

    if test == False:
        df = pd.read_csv('./datasets/classification/mv.csv')
        print(tpot_class(df))
    else:
        #CLASSIFICAZIONE
        for d in list_calss:
            df = pd.read_csv(d)
            res_class.update({d: []})
            print("1--------------------------------" + d + "--------------------------------1")
            print(df.info())
            print("Accuracy: " + tpot_class(df))
            #res_class[d].append({"auto-sklearn": tpot_class(df)})
            print("2--------------------------------" + d + "--------------------------------2")

            #print(autoSklearn_class(df)) # ritorna l'accuracy -> funziona
            #print(autokeras_class(df)) # ritorna l'accuracy
            #print(h20_class(df)) 
            #print(mlbox_class(df)) 
            #print(tpot_class(df)) # ritorna l'accuracy -> funziona

    #print(res_class)


        #REGRESSIONE
        #datasets = [f for f in listdir('../dataset/regression/') if isfile(join('../dataset/clasification/', f))]
        #for dt in datasets:
        #    df = pd.read_csv(dt)
        #    print(autoSklearn(df))
        #    print(autokeras(df))
        #    print(h20(df))
        #    print(mlbox(df))
        #    print(tpot(df))


        #stampo i grafici per confrontare i risultati