import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os
from datetime import datetime
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon
from utils.usefull_functions import scatter, hist
from utils.algo_functions import autogluon

datasets = [('titanic', 'classification'), ('tabular-playground-series-mar-2021', 'classification')]

def get_task(df):
    for d in datasets:
        if df == d[0]:
            return d[1]
    return False

def kaggle_benchmark(list_df):
    api = KaggleApi()
    api.authenticate()


    res_class = {'dataset': [], 'autosklearn-acc': [], 'autosklearn-f1': [], 'tpot-acc': [], 'tpot-f1': [], 'autokeras-acc': [], 'autokeras-f1': [], 'h2o-acc': [], 'h2o-f1': [], 'autogluon-acc': [], 'autogluon-f1': []}
    res_reg = {'dataset': [], 'autosklearn-rmse': [], 'autosklearn-r2': [], 'tpot-rmse': [], 'tpot-r2': [], 'autokeras-rmse': [], 'autokeras-r2': [], 'h2o-rmse': [], 'h2o-r2': [], 'autogluon-rmse': [], 'autogluon-r2': []}

    res_class = pd.DataFrame(res_class)
    res_reg = pd.DataFrame(res_reg)


    for df in list_df:
        task = get_task(df)
        if task is not False:
            print('------------------Dataset name: ' + df + ' - Task: ' + task + '------------------')
            path = './datasets/kaggle/' + df

            api.competition_download_files(df, path=path)

            zf = ZipFile(path + '/' + df + '.zip')
            zf.extractall(path) #save files in selected folder
            zf.close()

            os.remove(path + '/' + df + '.zip')

            train = pd.read_csv(path + '/train.csv')
            test = pd.read_csv(path + '/test.csv')


            res_as = fun_autosklearn((train, test), task)
            res_t = fun_tpot((train, test), task)
            res_ak = fun_autokeras((train, test), task)
            res_h = fun_h2o((train, test), task)
            res_ag = fun_autogluon((train, test), task)

            if(task == 'classification'):
                new_row = {'dataset': df, 'autosklearn-acc': res_as[0], 'autosklearn-f1': res_as[1], 'tpot-acc': res_t[0], 'tpot-f1': res_t[1], 'autokeras-acc': res_ak[0], 'autokeras-f1': res_ak[1], 'h2o-acc': res_h[0], 'h2o-f1': res_h[0], 'autogluon-acc': res_ag[0], 'autogluon-f1': res_ag[1]}
                res_class = res_class.append(new_row, ignore_index=True)
            else:
                new_row = {'dataset': df, 'autosklearn-rmse': res_as[0], 'autosklearn-r2': res_as[1], 'tpot-rmse': res_t[0], 'tpot-r2': res_t[1], 'autokeras-rmse': res_ak[0], 'autokeras-r2': res_ak[1], 'h2o-rmse': res_h[0], 'h2o-r2': res_h[0], 'autogluon-rmse': res_ag[0], 'autogluon-r2': res_ag[1]}
                res_reg = res_reg.append(new_row, ignore_index=True)

        else:
            print('\nDatasek di kaggle inesistente. Se esistente accertarsi di aver accettato le condizioni della competizione.\n')


    path = './results/kaggle/' + str(datetime.now())
    os.makedirs(path)
    if(not res_class.empty):
        print('---------------------------------RISULTATI DI CLASSIFICAZIONE KAGGLE---------------------------------')
        print(res_class)

        res_class.to_csv(path + '/classification.csv', index = False)
        hist(res_class, 'Kaggle - Classificazione')

    if(not res_reg.empty):
        print('\n\n---------------------------------RISULTATI DI REGRESSIONE KAGGLE---------------------------------')
        print(res_reg)

        res_reg.to_csv(path + '/regression.csv', index = False)
        hist(res_reg, 'Kaggle - Regressione')

    

