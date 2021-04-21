import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os
from datetime import datetime
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon
from utils.usefull_functions import scatter

datasets = [('titanic', 'classification'), ('tabular-playground-series-mar-2021', 'classification')]

def get_task(df):
    for d in datasets:
        if df == d[0]:
            return d[1]
    return False

def kaggle_benchmark(list_df):
    api = KaggleApi()
    api.authenticate()


    res_class = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'autogluon': [], 'best': []}
    res_reg = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'autogluon': [], 'best': []}

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


            res = [fun_autosklearn((train, test), task), fun_tpot((train, test), task), fun_autokeras((train, test), task), fun_h2o((train, test), task), fun_autogluon((train, test), task)]

            if(task == 'classification'):
                new_row = {'dataset': df, 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmax(res)+1] }
                res_class = res_class.append(new_row, ignore_index=True)
            else:
                new_row = {'dataset': df, 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmin(res)+1] }
                res_reg = res_reg.append(new_row, ignore_index=True)
        else:
            print('\nDatasek di kaggle inesistente. Se esistente accertarsi di aver accettato le condizioni della competizione.\n')


    path = './results/kaggle/' + str(datetime.now())
    os.makedirs(path)
    if(not res_class.empty):
        print('---------------------------------RISULTATI DI CLASSIFICAZIONE KAGGLE---------------------------------')
        print(res_class)

        res_class.to_csv(path + '/classification.csv', index = False)
        scatter(res_class, 'Kaggle - Classificazione')

    if(not res_reg.empty):
        print('\n\n---------------------------------RISULTATI DI REGRESSIONE KAGGLE---------------------------------')
        print(res_reg)

        res_reg.to_csv(path + '/regression.csv', index = False)
        scatter(res_reg, 'Kaggle - Regressione')

    

