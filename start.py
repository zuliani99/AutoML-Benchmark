#!/usr/bin/env python3

# AUTOKERAS E AUTOGLUON -> scrivono nel disco per salvare tutti i modelli
# Ora tutti gli algoritmi sono parallelizzati, utilizzano tutti i core disponibili

import sys


from sklearn.datasets import fetch_openml
from datetime import datetime
import os
import numpy as np
import pandas as pd
import openml



from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile



def fun_autosklearn(df, task):
    res_autosklearn = 0.0
    print("--------------------------------AUTOSKLEARN--------------------------------")
    try:
        res_autosklearn = (auto_sklearn(df, task))
        print('Risultato memorizzato!')
        return res_autosklearn
    except:
        print('Qualcosa è andato storto :(')
    print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
    return res_autosklearn


def fun_tpot(df, task):
    res_tpot = 0.0
    print("-----------------------------------TPOT------------------------------------")
    try:
        res_tpot = (TPOT(df, task))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("-----------------------------------TPOT------------------------------------\n\n")
    return res_tpot


def fun_autokeras(df, task):
    res_autokeras = 0.0
    print("---------------------------------AUTOKERAS---------------------------------")
    try:
        res_autokeras = (autokeras(df, task))[1]
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("---------------------------------AUTOKERAS---------------------------------\n\n")
    return res_autokeras

def fun_h2o(df, task):
    res_h2o = 0.0
    print("------------------------------------H2O------------------------------------")
    try:
        res_h2o = (H2O(df, task))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("------------------------------------H2O------------------------------------\n\n")
    return res_h2o


def fun_autogluon(df, task):
    res_autogluon = 0.0
    print("----------------------------------AUTOGLUON--------------------------------")
    try:
        res_autogluon = (autogluon(df, task))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("----------------------------------AUTOGLUON--------------------------------\n\n")
    return res_autogluon


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




def test():
    task = 'regression'

    id = 344
    X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
    y = y.to_frame()
    X[y.columns[0]] = y
    df = X

    print(auto_sklearn(df, task))
    '''res = [fun_autosklearn(df, task), fun_tpot(df, task), fun_autokeras(df, task), fun_h2o(df), fun_autogluon(d, task)]

    new_row = {'dataset': str(id) + '.csv', 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmax(res)+1] }


    res_class = res_class.append(new_row, ignore_index=True)
    print(res_class)

    res_reg = res_reg.append(new_row, ignore_index=True)
    print(res_reg)'''
        
    # autosklearn -> root_mean_squared_error: 0.7424714465226021
    # tpot -> reg:squarederror: -120.33836282299288
    # autokeras -> mean_squared_error: 0.006891193334013224 -> fixato
    # h2o -> mean_squared_error: 0.11184497546233797 -> fixato
    # autogluon -> root_mean_squared_error: 0.029061951526943217


def kaggle_benchmark():
    api = KaggleApi()
    api.authenticate()

    dataset = 'titanic'
    path = './datasets/kaggle/' + dataset

    api.competition_download_files(dataset, path=path)

    zf = ZipFile(path + '/' + dataset + '.zip')
    zf.extractall(path) #save files in selected folder
    zf.close()

    os.remove(path + '/' + dataset + '.zip')

    train = pd.read_csv(path + '/train.csv')
    test = pd.read_csv(path + '/test.csv')

    print(train.head())
    print(test.head())


def main():
    print("---------------------------------------START---------------------------------------")

    try:
        param = (sys.argv[1])
        if param.isnumeric():
            if int(param) > 0:
                openml_benchmark(int(param))
            else:
                print('Inserisci un numero positivo oppure non inserire nulla per eseguire un test singolo')
        else:
            if param =='kaggle':
                kaggle_benchmark()
            else:
                print('Comando non valido!')
    except IndexError:
        test()


if __name__ == '__main__':
    from algorithms.auto_sklearn import auto_sklearn
    from algorithms.tpot import TPOT
    from algorithms.h2o import H2O
    from algorithms.auto_keras import autokeras
    from algorithms.auto_gluon import autogluon
    main()
        