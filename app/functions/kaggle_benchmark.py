# Import necessari
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from utils.result_class import Result
import os

# Dataframe provenienti da competizioni Kaggle
dataframes = {
    'titanic': {'task': 'classification', 'measure': 'acc'},
    'contradictory-my-dear-watson': {'task': 'classification', 'measure': 'acc'},
    'forest-cover-type-prediction': {'task': 'classification', 'measure': 'acc'},
    'ghouls-goblins-and-ghosts-boo': {'task': 'classification', 'measure': 'acc'},

    'commonlitreadabilityprize': {'task': 'regression', 'measure': 'rmse'},
    'bigquery-geotab-intersection-congestion': {'task': 'regression', 'measure': 'rmse'},
    'tabular-playground-series-feb-2021': {'task': 'regression', 'measure': 'rmse'},
    'GEF2012-wind-forecasting': {'task': 'regression', 'measure': 'rmse'}, 
}

# Funzione addetta ad effettuare un aggiuntivo unzip in caso sia necessario
def unzip_more(file_extracted, path):
    for i, file in enumerate(file_extracted):
        splitted = file.split('.')
        if(splitted[len(splitted)-1] == 'zip'):
            zf = ZipFile(path + '/' + file)
            zf.extractall(path) 
            os.remove(path  + '/' + file)
            zf.close()

# Funzione per l'ottenimento del miglio risultato proveninete dalla leaderboard del rispettivo DataFrame di una competizione
def get_leader(leaderboard):
    i = 0
    leader = leaderboard['submissions'][i]
    while float(leaderboard['submissions'][i]['score']) <= 0.0:
        i+=1
        leader = leaderboard['submissions'][i]
    return leader

# Funzione per l'esecuzione del Kaggle Benchmark
def kaggle_benchmark(list_df, options):
    # Inizializzazione dell'API Kaggle
    api = KaggleApi()
    api.authenticate()

    res_kaggle = Result('Kaggle') # Inizializzazione di un nuove Result di tipo Kaggle
    if not isinstance(list_df, list): list_df = [list_df]
    for df in list_df: # Per tutti i DataFrame che l'utente ha selto
        print('------------------Dataset name: ' + df + ' - Task: ' + dataframes[df]['task'] + '------------------')
        path = './dataframes/Kaggle/' + df

        if(not os.path.isdir(path)): # Se il DataFrame non è presente devo scaricarlo e unzipparlo
            api.competition_download_files(df, path=path) 

            # Estraggo le cartelle di train test e submission
            zf = ZipFile(path + '/' + df + '.zip')
            zf.extractall(path) 
            zf.close()
            
            # Rimuovo lo zip file
            os.remove(path + '/' + df + '.zip')

            file_extracted = (os.listdir(path))
            unzip_more(file_extracted, path)

        train = pd.read_csv(path + '/train.csv')
        test = pd.read_csv(path + '/test.csv')

        leaderboard = api.competition_view_leaderboard(df) # Download della leaderboard della competizione Kaggle
        leader = get_leader(leaderboard) # ottendo il Leader
            
        res_kaggle.run_benchmark((train, test), dataframes[df]['task'], df, {'measure': dataframes[df]['measure'], 'score': leader['score']}, options)
    return res_kaggle.print_res()