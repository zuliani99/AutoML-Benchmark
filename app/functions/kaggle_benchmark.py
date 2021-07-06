import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from utils.result_class import Result
import os

datasets = {
    'titanic': {'task': 'classification', 'measure': 'acc'},
    'contradictory-my-dear-watson': {'task': 'classification', 'measure': 'acc'},
    'forest-cover-type-prediction': {'task': 'classification', 'measure': 'acc'},
    'ghouls-goblins-and-ghosts-boo': {'task': 'classification', 'measure': 'acc'},
    'sentiment-analysis-on-movie-reviews': {'task': 'classification', 'measure': 'acc'},

    'commonlitreadabilityprize': {'task': 'regression', 'measure': 'rmse'},
    'bigquery-geotab-intersection-congestion': {'task': 'regression', 'measure': 'rmse'},
    'restaurant-revenue-prediction': {'task': 'regression', 'measure': 'rmse'},
    'tabular-playground-series-feb-2021': {'task': 'regression', 'measure': 'rmse'},
    'predict-citations-for-us-patents': {'task': 'regression', 'measure': 'r2'},
}

def unzip_more(file_extracted, path):
    for i, file in enumerate(file_extracted):
        splitted = file.split('.')
        if(splitted[len(splitted)-1] == 'zip'):
            zf = ZipFile(path + '/' + file)
            zf.extractall(path) 
            os.remove(path  + '/' + file)
            zf.close()

def get_leader(leaderboard):
    i = 0
    leader = leaderboard['submissions'][i]
    while float(leaderboard['submissions'][i]['score']) <= 0.0:
        i+=1
        leader = leaderboard['submissions'][i]
    return leader

def kaggle_benchmark(list_df, options):
    api = KaggleApi()
    api.authenticate()

    res_kaggle = Result('Kaggle')
    if not isinstance(list_df, list): list_df = [list_df]
    for df in list_df:
            print('------------------Dataset name: ' + df + ' - Task: ' + datasets[df]['task'] + '------------------')
            path = './datasets/Kaggle/' + df

            api.competition_download_files(df, path=path)

            #estraggo le cartelle di train test e submission
            zf = ZipFile(path + '/' + df + '.zip')
            zf.extractall(path) 
            zf.close()

            os.remove(path + '/' + df + '.zip')

            file_extracted = (os.listdir(path))
            unzip_more(file_extracted, path)

            train = pd.read_csv(path + '/train.csv')
            test = pd.read_csv(path + '/test.csv')

            leaderboard = api.competition_view_leaderboard(df)
            leader = get_leader(leaderboard)
            
            res_kaggle.run_benchmark((train, test), datasets[df]['task'], df, {'measure': datasets[df]['measure'], 'score': leader['score']}, options)
    return res_kaggle.print_res()