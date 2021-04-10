#!/usr/bin/env python3

import openml
import os.path
import os
import pandas as pd
import numpy as np
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.datasets import fetch_openml
from os import listdir
from os.path import isfile



#from algorithms.auto_keras import autokeras_class
#from algorithms.auto_gluon import autogluon
from algorithms.tpot import tpot_class
from algorithms.h2o import h2o_class
from algorithms.auto_sklearn import autoSklearn_class




def fun_autosklearn(df):
    res_autosklearn = 0.0
    print("--------------------------------AUTOSKLEARN--------------------------------")
    try:
        res_autosklearn = (autoSklearn_class(df))
        print('Risultato memorizzato!')
        return res_autosklearn
    except:
        print('Qualcosa è andato storto :(')
    print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
    return res_autosklearn


def fun_tpot(df):
    res_tpot = 0.0
    print("-----------------------------------TPOT------------------------------------")
    try:
        res_tpot = (tpot_class(df))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("-----------------------------------TPOT------------------------------------\n\n")
    return res_tpot


def fun_autokeras(df):
    from algorithms.auto_keras import autokeras_class
    res_autokeras = 0.0
    print("---------------------------------AUTOKERAS---------------------------------")
    try:
        res_autokeras = (autokeras_class(df))[1]
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("---------------------------------AUTOKERAS---------------------------------\n\n")
    return res_autokeras

def fun_h2o(df):
    res_h2o = 0.0
    print("------------------------------------H2O------------------------------------")
    try:
        res_h2o = (h2o_class(df))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("------------------------------------H2O------------------------------------\n\n")
    return res_h2o


def fun_autogluon(df):
    from algorithms.auto_gluon import autogluon
    res_autogluon = 0.0
    print("----------------------------------AUTOGLUON--------------------------------")
    try:
        res_autogluon = (autogluon(df))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("----------------------------------AUTOGLUON--------------------------------\n\n")
    return res_autogluon



def main():
    print("---------------------------------------START---------------------------------------")

      
    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > 40000].sort_values(["NumberOfInstances"]).head(5)

    df_good = 0
    df_bad = 0

    list_class = []
    list_reg = []

    res_class = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'autogluon': [], 'best': []}
    res_reg = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'autogluon': [], 'best': []}

    res_class = pd.DataFrame(res_class)
    res_reg = pd.DataFrame(res_reg)

    test = True

    if test == False:
        print('--------------------------------Inizio Dataset Download--------------------------------')
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
            print('--------------------------------Fine Dataset Download--------------------------------')

        print('Good df: ' + str(df_good) + '    bad df: ' + str(df_bad) + '\n')


        #CLASSIFICAZIONE
        for d in list_class:
            df = pd.read_csv(d)
            
            print('---------------------------------Dataset: ' + d + '---------------------------------\n')

            res = [fun_autosklearn(df), fun_tpot(df), fun_autokeras(df), fun_h2o(df), fun_autogluon('./datasets/classification/' + d.split('/')[3])]

            new_row = {'dataset': d.split('/')[3], 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmax(res)+1] }

            res_class = res_class.append(new_row, ignore_index=True)
        print('---------------------------------RISULTATI DI CLASSIFICAZIONE---------------------------------')
        print(res_class)


        #REGRESSIONE
        #for d in list_reg:
            #df = pd.read_csv(d)

            
        
    else:
        id = 727
        X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
        y = y.to_frame()
        X[y.columns[0]] = y
        df = X

        res = [fun_autosklearn(df), fun_tpot(df), fun_autokeras(df), fun_h2o(df), fun_autogluon('./datasets/classification/' + str(id) + '.csv')]

        new_row = {'dataset': str(id) + '.csv', 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmax(res)+1] }

        res_class = res_class.append(new_row, ignore_index=True)

        print(res_class)

        #print(fun_autogluon(df))


if __name__ == '__main__':  
    main()
        
