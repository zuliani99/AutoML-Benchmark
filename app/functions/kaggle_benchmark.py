import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from utils.result_class import Result
import os

datasets = [
    ('titanic', 'classification'),
    ('tabular-playground-series-mar-2021', 'classification'),
    ('mercedes-benz-greener-manufacturing', 'regression'),
    ('restaurant-revenue-prediction', 'regression')
]

def kaggle_benchmark(list_df):
    api = KaggleApi()
    api.authenticate()

    res_kaggle = Result('Kaggle')
    if not isinstance(list_df, list):
        list_df = [list_df]
    for df in list_df:
        if (df, 'classification') in datasets or (df, 'regression') in datasets:
            task = 'classification' if (df, 'classification') in datasets else 'regression'
            print('------------------Dataset name: ' + df + ' - Task: ' + task + '------------------')
            path = './datasets/Kaggle/' + df

            api.competition_download_files(df, path=path)

            #estraggo le cartelle di train test e submission
            zf = ZipFile(path + '/' + df + '.zip')
            zf.extractall(path) 
            zf.close()

            os.remove(path + '/' + df + '.zip')

            file_extracted = (os.listdir(path))
            #print(file_extracted)

            for file in file_extracted:
                #print(file, is_zipfile(file))
                splitted = file.split('.')
                #print(splitted)
                if(splitted[len(splitted)-1] == 'zip'):
                    #print('sono dentro')
                    zf = ZipFile(path + '/' + file)
                    zf.extractall(path) 
                    os.remove(path  + '/' + file)
            zf.close()

            train = pd.read_csv(path + '/train.csv')
            test = pd.read_csv(path + '/test.csv')

            leaderboard = api.competition_view_leaderboard(df)
            '''
            ritorna la leaderboard da capire come la ritorna
            devo inserire il migliore nella tabella risultante così posso vedere come sono andati gli altri algoritmi
            '''

            print('STO PER STAMPARE LACLASSFICA DEL DATASET')
            leader = leaderboard['submissions'][0]
            print(leader['teamName'], leader['score'])

            res_kaggle.run_benchmark((train, test), task, df, {'name': leader['teamName'], 'score': leader['score']})
        else:
            print('\nDataset kaggle "'+ df +'" inesistente. Se esistente accertarsi di aver accettato le condizioni della competizione.\n')
            return 'Dataset kaggle "'+ df +'" inesistente. Se esistente accertarsi di aver accettato le condizioni della competizione.'
    return res_kaggle.print_res()