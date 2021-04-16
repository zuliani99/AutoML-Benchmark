import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os
from algorithms.auto_sklearn import auto_sklearn_k

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

    print(auto_sklearn_k(train, test, 'classification'))
    

