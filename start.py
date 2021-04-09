#!/usr/bin/env python3

import openml
import os.path
import os
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.datasets import fetch_openml
from os import listdir
from os.path import isfile


from algorithms.auto_sklearn import autoSklearn_class
from algorithms.tpot import tpot_class
from algorithms.auto_keras import autokeras_class
from algorithms.h2o import h2o_class
from algorithms.ludwig import ludwig_class



def main():
    print("--------------------START--------------------")
      
    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > 40000].sort_values(["NumberOfInstances"]).head(3)

    df_good = 0
    df_bad = 0

    list_class = []
    list_reg = []

    res_class = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'ludwig': []}
    res_reg = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'ludwig': []}

    res_class = pd.DataFrame(res_class)
    res_reg = pd.DataFrame(res_reg)

    test = True

    if test == False:

        for row in df_id_name.iterrows():
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

            new_row = {'dataset': d, 
                        'autosklearn': autoSklearn_class(df), 
                        'tpot': tpot_class(df), 
                        'autokeras': autokeras_class(df)[1], 
                        'h2o': h2o_class(df), 
                        'ludwig': ludwig_class(df)[2] }

            res_calss = res_calss.append(new_row, ignore_index=True)

            
        
    else:
        id = 727
        X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
        y = y.to_frame()
        X[y.columns[0]] = y
        df = X


        print("--------------------------------AUTOSKLEARN--------------------------------")
        res_autosklearn = (autoSklearn_class(df))# -> non da errore di tensorflow
        print("--------------------------------AUTOSKLEARN--------------------------------\n\n")


        print("-----------------------------------TPOT------------------------------------")
        res_tpot = (tpot_class(df)) # non da errore di tensorflow, con gli import fino a qui
        print("-----------------------------------TPOT------------------------------------\n\n")


        print("---------------------------------AUTOKERAS---------------------------------")
        res_autokeras = (autokeras_class(df))[1] # dc qua lo darà sicuramente 
        print("---------------------------------AUTOKERAS---------------------------------\n\n")


        print("------------------------------------H2O------------------------------------")
        res_h2o = (h2o_class(df))
        print("------------------------------------H2O------------------------------------\n\n")


        print("-----------------------------------LUDWIG----------------------------------")
        res_ludwig = (ludwig_class(df))[2]
        print("-----------------------------------LUDWIG----------------------------------\n\n")
        
        new_row = {'dataset': id, 'autosklearn': res_autosklearn, 'tpot': res_tpot, 'autokeras': res_autokeras, 'h2o': res_h2o, 'ludwig': res_ludwig}

        res_calss = res_calss.append(new_row, ignore_index=True)

        print(res_calss)



if __name__ == '__main__':  
    main()
        
