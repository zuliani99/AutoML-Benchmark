from algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon
from sklearn.datasets import fetch_openml
from datetime import datetime
import os
import numpy as np
import pandas as pd
import openml

def openml_benchmark(df_n):
    df_good = 0
    df_bad = 0

    list_df = []

    res_class = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'autogluon': [], 'best': []}
    res_reg = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'autogluon': [], 'best': []}

    res_class = pd.DataFrame(res_class)
    res_reg = pd.DataFrame(res_reg)


    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > 40000].sort_values(["NumberOfInstances"]).head(df_n)


    print('--------------------------------Inizio Dataset Download--------------------------------')
    for index, row in df_id_name.iterrows():
        print('------------------Dataset ID: ' + str(row['did']) + ' name: ' + row['name'] + '------------------')
        try:
            X, y = fetch_openml(data_id=row['did'], as_frame=True, return_X_y=True)
            y = y.to_frame()

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

            list_df.append(fullname)

        except:
            print("bad df\n")
            df_bad+=1
        print('--------------------------------Fine Dataset Download--------------------------------')

    print('Good df: ' + str(df_good) + '    bad df: ' + str(df_bad) + '\n')
    



    #ESECUZUONE DEGLI ALGORITMI
    for d in list_df:
        task = d.split('/')[2]
        df = pd.read_csv(d)
                
        print('---------------------------------Dataset: ' + d + '---------------------------------\n')

        res = [fun_autosklearn(df, task), fun_tpot(df, task), fun_autokeras(df, task), fun_h2o(df, task), fun_autogluon(d, task)]

        if(task == 'classification'):
            new_row = {'dataset': d.split('/')[3], 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmax(res)+1] }
            res_class = res_class.append(new_row, ignore_index=True)
        else:
            new_row = {'dataset': d.split('/')[3], 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmin(res)+1] }
            res_reg = res_reg.append(new_row, ignore_index=True)

    print('---------------------------------RISULTATI DI CLASSIFICAZIONE---------------------------------')
    print(res_class)
    print('\n\n---------------------------------RISULTATI DI REGRESSIONE---------------------------------')
    print(res_reg)

    path = './results/' + str(datetime.now())
    os.makedirs(path)
    res_class.to_csv(path + '/classification.csv', index = False)
    res_reg.to_csv(path + '/regression.csv', index = False)