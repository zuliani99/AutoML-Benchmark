#pip3 install openml

import openml
import os.path
import os
import pandas as pd
import numpy as np
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.datasets import fetch_openml
from os import listdir
from os.path import isfile, join



#from algorithms.classification.auto_sklearn import autoSklearn_class
#from algorithms.classification.auto_keras import autokeras_class
#from algorithms.classification.h2o import h2o_class
#from algorithms.classification.mlbox import mlbox_class
#from algorithms.classification.tpot import tpot_class
from algorithms.classification.ludwig import ludwig_class



if __name__ == '__main__':   

    print("--------------------START--------------------")
    
    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > 40000].sort_values(["NumberOfInstances"]).head(5)

    df_good = 0
    df_bad = 0

    list_class = []
    list_reg = []

    res_class = {}
    res_res = {}

    test = True

    #print(df_id_name.info())

    if test == False:

        for index, row in df_id_name.iterrows():
            print('------------------Dataset ID: ' + str(row['did']) + ' name: ' + row['name'] + '------------------')
            try:
                X, y = fetch_openml(data_id=row['did'], as_frame=True, return_X_y=True)
                y = y.to_frame()

                print(y.head())
                print(y.info())

                if pd.api.types.infer_dtype(y[y.columns[0]]) == "categorical" or pd.api.types.infer_dtype(y[y.columns[0]]) == "boolean":
                    file_dir =  './datasets/classification/'
                else:
                    file_dir =  './datasets/regression/'
                    
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                fullname = os.path.join(file_dir, str(row['did']) + '.csv')

                print("good df " + fullname + '\n')
                df_good+=1

                if not os.path.exists(fullname):
                    X[y.columns[0]] = y
                    X.to_csv(fullname, index=False, header=True)

                if(file_dir == './datasets/regression/'):
                    list_reg.append(fullname)
                else:
                    list_class.append(fullname)

            except:
                print("bad df\n")
                df_bad+=1
            print('------------------------------------')

        print('Good df: ' + str(df_good) + '    bad df: ' + str(df_bad) + '\n')

        print(list_class)
        print(list_reg)

        #CLASSIFICAZIONE
        for d in list_class:
            df = pd.read_csv(d)
            #res_class.update({d: []})
            print("1--------------------------------" + d + "--------------------------------1")
            print(df.info())
            #print("Accuracy: " + ludwig_class(df))
            res = ludwig_class(df)
            print("Beat epoch: " + str(res[0]) + "   Validation accuracy: " + str(res[1]) + "     train accuracy: " + str(res[2])
            #res_class[d].append({"auto-sklearn": tpot_class(df)})
            print("2--------------------------------" + d + "--------------------------------2")


            #print(autoSklearn_class(df)) # ritorna l'accuracy -> funziona
            #print(autokeras_class(df)) # ritorna l'accuracy
            #print(h20_class(df)) 
            #print(mlbox_class(df)) 
            #print(tpot_class(df)) # ritorna l'accuracy -> funziona



    else:

        X, y = fetch_openml(data_id=727, as_frame=True, return_X_y=True, cache=True)
        y = y.to_frame()
        X[y.columns[0]] = y
        df = X
        #df = pd.read_csv('./datasets/classification/mv.csv')
        print(ludwig_class(df))
