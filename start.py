#!/usr/bin/env python3

import openml
import os.path
import os
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.datasets import fetch_openml
from os import listdir
from os.path import isfile


from algorithms.ludwig import ludwig_class
from algorithms.auto_sklearn import autoSklearn_class
from algorithms.tpot import tpot_class
from algorithms.auto_keras import autokeras_class
from algorithms.h2o import h2o_class


import tensorflow as tf
from tensorflow.python.eager import context


def fun_autosklearn(df):
    print("--------------------------------AUTOSKLEARN--------------------------------")
    res_autosklearn = 0.0
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
    #try:
    res_tpot = (tpot_class(df))
    print('Risultato memorizzato!')
    #except:
    print('Qualcosa è andato storto :(')
    print("-----------------------------------TPOT------------------------------------\n\n")
    return res_tpot


def fun_autokeras(df):
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


def fun_ludwig(df):
    res_ludwig = 0.0
    print("-----------------------------------LUDWIG----------------------------------")
    try:
        res_ludwig = (ludwig_class(df))[2] # -> RuntimeError: Intra op parallelism cannot be modified after initialization.
        print('Risultato memorizzato!')
    except RuntimeError:
        print('Intra op parallelism cannot be modified after initialization.')
    except:
        print('Qualcosa è andato storto :(')
    print("-----------------------------------LUDWIG----------------------------------\n\n")
    return res_ludwig



def main():
    print("---------------------------------------START---------------------------------------")

    #config tensorflow set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    _ = tf.Variable([1])

      
    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > 40000].sort_values(["NumberOfInstances"]).head(7)

    df_good = 0
    df_bad = 0

    list_class = []
    list_reg = []

    res_class = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'ludwig': []}
    res_reg = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'ludwig': []}

    res_class = pd.DataFrame(res_class)
    res_reg = pd.DataFrame(res_reg)

    test = False

    if test == False:
        print('--------------------------------Dataset download--------------------------------')
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
            
            print('---------------------------------Dataset: ' + d + '---------------------------------\n')

            new_row = {'dataset': d.split('/')[3], 'autosklearn': fun_autosklearn(df),
                        'tpot': fun_tpot(df),
                        'autokeras': fun_autokeras(df),
                        'h2o': fun_h2o(df),
                        'ludwig': fun_ludwig(df) }

            res_class = res_class.append(new_row, ignore_index=True)

        print(res_class)
            
        
    else:
        id = 881
        X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
        y = y.to_frame()
        X[y.columns[0]] = y
        df = X

        '''new_row = {'dataset': id, 'autosklearn': fun_autosklearn(df),
                        'tpot': fun_tpot(df),
                        'autokeras': fun_autokeras(df),
                        'h2o': fun_h2o(df),
                        'ludwig': fun_ludwig(df) }

        res_class = res_class.append(new_row, ignore_index=True)

        print(res_class)'''

        #print(fun_tpot(df))


if __name__ == '__main__':  
    main()
        
