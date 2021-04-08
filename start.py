#!/usr/bin/env python3

import openml
import os.path
import os
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.datasets import fetch_openml
from os import listdir
from os.path import isfile


def autosklearn_function(df):
    print("--------------------------------AUTOSKLEARN--------------------------------")
    from algorithms.auto_sklearn import autoSklearn_class
    res = (autoSklearn_class(df))
    print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
    return res

def tpot_function(df):
    print("--------------------------------TPOT--------------------------------")
    from algorithms.tpot import tpot_class
    res = (tpot_class(df))
    print("--------------------------------TPOT--------------------------------\n\n")
    return res

def autokeras_function(df):
    print("--------------------------------AUTOKERAS--------------------------------")
    from algorithms.auto_keras import autokeras_class
    res = (autokeras_class(df))
    print("--------------------------------AUTOKERAS--------------------------------\n\n")
    return res

def h2o_function(df):
    print("--------------------------------H2O--------------------------------") 
    from algorithms.h2o import h2o_class
    res = (h2o_class(df))
    print("--------------------------------H2O--------------------------------\n\n")
    return res

def ludwig_function(df):
    print("--------------------------------LUDWIG--------------------------------")
    from algorithms.ludwig import ludwig_class
    res = (ludwig_class(df))
    print("--------------------------------LUDWIG--------------------------------\n\n")
    return res



if __name__ == '__main__':   
    print("--------------------START--------------------")
    
    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > 40000].sort_values(["NumberOfInstances"]).head(3)

    df_good = 0
    df_bad = 0

    list_class = []
    list_reg = []

    res_class = {}
    res_reg = {}

    test = True

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
                    res_class[row['did']] = []
                else:
                    file_dir =  './datasets/regression/'
                    res_reg[row['did']] = []
                    
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

            res_class[d].append({'autosklearn': autosklearn_function(df)})
            res_class[d].append({'tpot': tpot_function(df)})
            res_class[d].append({'autokeras': autokeras_function(df)})
            res_class[d].append({'h2o': h2o_function(df)})
            res_class[d].append({'ludwig': ludwig_function(df)})

            
            



    else:
        X, y = fetch_openml(data_id=727, as_frame=True, return_X_y=True, cache=True)
        y = y.to_frame()
        X[y.columns[0]] = y
        df = X

        d = str(727)

        #res_class[d].append({'autosklearn': autosklearn_function(df)})
        #res_class[d].append({'tpot': tpot_function(df)})
        #res_class[d].append({'autokeras': autokeras_function(df)})
        #res_class[d].append({'h2o': h2o_function(df)})
        #res_class[d].append({'ludwig': ludwig_function(df)})

        print(autosklearn_function(df))
        print(tpot_function(df))
        print(autokeras_function(df))
        print(h2o_function(df))
        print(ludwig_function(df))


        
        
