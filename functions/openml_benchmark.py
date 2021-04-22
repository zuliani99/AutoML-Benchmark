from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon
from utils.usefull_functions import scatter, hist
from sklearn.datasets import fetch_openml
from datetime import datetime
import os
import numpy as np
import pandas as pd
import openml

def openml_benchmark(df_n):
    n_class = df_n
    n_reg = df_n

    list_df = []

    res_class = {'dataset': [], 'autosklearn-acc': [], 'autosklearn-f1': [], 'tpot-acc': [], 'tpot-f1': [], 'autokeras-acc': [], 'autokeras-f1': [], 'h2o-acc': [], 'h2o-f1': [], 'autogluon-acc': [], 'autogluon-f1': []}
    res_reg = {'dataset': [], 'autosklearn-rmse': [], 'autosklearn-r2': [], 'tpot-rmse': [], 'tpot-r2': [], 'autokeras-rmse': [], 'autokeras-r2': [], 'h2o-rmse': [], 'h2o-r2': [], 'autogluon-rmse': [], 'autogluon-r2': []}

    res_class = pd.DataFrame(res_class)
    res_reg = pd.DataFrame(res_reg)


    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > 40000].sort_values(["NumberOfInstances"]) #.head(df_n)
    

    print('--------------------------------Inizio Dataset Download--------------------------------')
    

    #while n_class > 0 and n_reg > 0:
    for index, row in df_id_name.iterrows():
        try:
            if not os.path.exists('./datasets/classification/' + str(row['did']) + '.csv') and not os.path.exists('./datasets/regression/' + str(row['did']) + '.csv'):
                X, y = fetch_openml(data_id=row[0], as_frame=True, return_X_y=True, cache=True)
                if y is not None:
                    y = y.to_frame()
                else:
                    y = X.iloc[:, -1].to_frame()
                    X = X.drop(y.columns[0], axis=1)

                t = pd.api.types.infer_dtype(y[y.columns[0]])
                if (t == "categorical" or t == "boolean"):
                    file_dir =  './datasets/classification/'

                if (t == "floating" or t == 'integer' or t == 'decimal'):
                    file_dir =  './datasets/regression/'

                if n_class > 0 or n_reg > 0:
                    print('------------------Dataset ID: ' + str(row['did']) + ' name: ' + str(row['name']) + '------------------')
                    print(y.info())
                    fullname = os.path.join(file_dir, str(row['did']) + '.csv')

                    print("good df " + fullname + '\n')

                    X[y.columns[0]] = y
                    X.to_csv(fullname, index=False, header=True)

                    list_df.append(fullname)

                    if file_dir == './datasets/classification/':
                        n_class-=1
                    else:
                        n_reg-=1
            else:
                if os.path.exists('./datasets/classification/' + str(row['did']) + '.csv') and n_class > 0:
                    print('------------------Dataset ID: ' + str(row['did']) + ' name: ' + str(row['name']) + '------------------')
                    print('-------------------------Dataset già presente-------------------------\n')
                    list_df.append('./datasets/classification/' + str(row['did']) + '.csv')
                    n_class-=1
                if os.path.exists('./datasets/regression/' + str(row['did']) + '.csv') and n_reg > 0:
                    print('------------------Dataset ID: ' + str(row['did']) + ' name: ' + str(row['name']) + '------------------')
                    print('-------------------------Dataset già presente-------------------------\n')
                    list_df.append('./datasets/regression/' + str(row['did']) + '.csv')
                    n_reg-=1

        except:
            print("bad df\n")

        if n_class == 0 and n_reg == 0:
            print('--------------------------------Fine Dataset Download--------------------------------')
            break
    

    #ESECUZUONE DEGLI ALGORITMI
    for d in list_df:
        task = d.split('/')[2]
        df = pd.read_csv(d)
                
        print('---------------------------------Dataset: ' + d + '---------------------------------\n')
        res_as = fun_autosklearn(df, task)
        res_t = fun_tpot(df, task)
        res_h = fun_autokeras(df, task)
        res_ak = fun_h2o(df, task)
        res_ag = fun_autogluon(d, task)

        if(task == 'classification'):
            #new_row = {'dataset': , 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmax(res)+1] }
            new_row = {'dataset': d.split('/')[3], 'autosklearn-acc': res_as[0], 'autosklearn-f1': res_as[1], 'tpot-acc': res_t[0], 'tpot-f1': res_t[1], 'autokeras-acc': res_ak[0], 'autokeras-f1': res_ak[1], 'h2o-acc': res_h[0], 'h2o-f1': res_h[0], 'autogluon-acc': res_ag[0], 'autogluon-f1': res_ag[1]}
            res_class = res_class.append(new_row, ignore_index=True)
        else:
            #new_row = {'dataset': d.split('/')[3], 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmin(res)+1] }
            new_row = {'dataset': d.split('/')[3], 'autosklearn-rmse': res_as[0], 'autosklearn-r2': res_as[1], 'tpot-rmse': res_t[0], 'tpot-r2': res_t[1], 'autokeras-rmse': res_ak[0], 'autokeras-r2': res_ak[1], 'h2o-rmse': res_h[0], 'h2o-r2': res_h[0], 'autogluon-rmse': res_ag[0], 'autogluon-r2': res_ag[1]}
            res_reg = res_reg.append(new_row, ignore_index=True)


    print('---------------------------------RISULTATI DI CLASSIFICAZIONE OPENML---------------------------------')
    print(res_class)
    print('\n\n---------------------------------RISULTATI DI REGRESSIONE OPENML---------------------------------')
    print(res_reg)

    path = './results/openml/' + str(datetime.now())
    os.makedirs(path)
    if(not res_class.empty):
        res_class.to_csv(path + '/classification.csv', index = False)
        hist(res_class, 'OpenML - Classificazione')
    if(not res_reg.empty):
        res_reg.to_csv(path + '/regression.csv', index = False)
        hist(res_reg, 'OpenML - Regressione')