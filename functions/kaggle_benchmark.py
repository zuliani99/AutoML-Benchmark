import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os
from datetime import datetime
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon

def kaggle_benchmark():
    api = KaggleApi()
    api.authenticate()

    datasets = [('titanic', 'classification')]

    res_class = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'autogluon': [], 'best': []}
    res_reg = {'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [], 'autogluon': [], 'best': []}

    res_class = pd.DataFrame(res_class)
    res_reg = pd.DataFrame(res_reg)


    for dataset in datasets:
        print('------------------Dataset name: ' + dataset[0] + ' - Task: ' + dataset[1] + '------------------')
        path = './datasets/kaggle/' + dataset[0]

        api.competition_download_files(dataset[0], path=path)

        zf = ZipFile(path + '/' + dataset[0] + '.zip')
        zf.extractall(path) #save files in selected folder
        zf.close()

        os.remove(path + '/' + dataset[0] + '.zip')

        train = pd.read_csv(path + '/train.csv')
        test = pd.read_csv(path + '/test.csv')


        res = [fun_autosklearn((train, test), dataset[1]), fun_tpot((train, test), dataset[1]), fun_autokeras((train, test), dataset[1]), fun_h2o((train, test), dataset[1]), fun_autogluon((train, test), dataset[1])]

        if(dataset[1] == 'classification'):
            new_row = {'dataset': dataset[0], 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmax(res)+1] }
            res_class = res_class.append(new_row, ignore_index=True)
        else:
            new_row = {'dataset': dataset[0], 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmin(res)+1] }
            res_reg = res_reg.append(new_row, ignore_index=True)


    print('---------------------------------RISULTATI DI CLASSIFICAZIONE KAGGLE---------------------------------')
    print(res_class)
    print('\n\n---------------------------------RISULTATI DI REGRESSIONE KAGGLE---------------------------------')
    print(res_reg)

    path = './results/kaggle/' + str(datetime.now())
    os.makedirs(path)
    if(not res_class.empty):
        res_class.to_csv(path + '/classification.csv', index = False)
    if(not res_reg.empty):
        res_reg.to_csv(path + '/regression.csv', index = False)

    

