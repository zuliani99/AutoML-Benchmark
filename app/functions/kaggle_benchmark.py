# Import needed
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from utils.result_class import Result
import os

# Dataframes from Kaggle competitions
dataframes = {
    'titanic': {'task': 'classification', 'measure': 'acc'},
    'contradictory-my-dear-watson': {'task': 'classification', 'measure': 'acc'},
    'forest-cover-type-prediction': {'task': 'classification', 'measure': 'acc'},
    'ghouls-goblins-and-ghosts-boo': {'task': 'classification', 'measure': 'acc'},

    'commonlitreadabilityprize': {'task': 'regression', 'measure': 'rmse'},
    'tabular-playground-series-jan-2021': {'task': 'regression', 'measure': 'rmse'},
    'mercedes-benz-greener-manufacturing': {'task': 'regression', 'measure': 'r2'},
    'GEF2012-wind-forecasting': {'task': 'regression', 'measure': 'rmse'}, 
}

# Function responsible for making an additional unzip if necessary
def unzip_more(file_extracted, path):
    for i, file in enumerate(file_extracted):
        splitted = file.split('.')
        if(splitted[len(splitted)-1] == 'zip'):
            zf = ZipFile(path + '/' + file)
            zf.extractall(path) 
            os.remove(path  + '/' + file)
            zf.close()

# Function for obtaining the best result from the leaderboard of the respective DataFrame of a competition
def get_leader(leaderboard):
    i = 0
    leader = leaderboard['submissions'][i]
    while float(leaderboard['submissions'][i]['score']) <= 0.0:
        i+=1
        leader = leaderboard['submissions'][i]
    return leader

# Function for running the Kaggle Benchmark
def kaggle_benchmark(list_df, options):
    # Initializing the Kaggle API
    api = KaggleApi()
    api.authenticate()

    res_kaggle = Result('Kaggle') # Initialization of a new Result of type Kaggle
    if not isinstance(list_df, list): list_df = [list_df]
    for df in list_df: # For all DataFrames that the user has chosen
        print('------------------Dataset name: ' + df + ' - Task: ' + dataframes[df]['task'] + '------------------')
        path = './dataframes/Kaggle/' + df

        if(not os.path.isdir(path)): # If the DataFrame is not present I have to download and unzip it
            api.competition_download_files(df, path=path) 

            # I extract the train test and submission folders
            zf = ZipFile(path + '/' + df + '.zip')
            zf.extractall(path) 
            zf.close()
            
            # Remove the zip file
            os.remove(path + '/' + df + '.zip')

            file_extracted = (os.listdir(path))
            unzip_more(file_extracted, path)

        train = pd.read_csv(path + '/train.csv')
        test = pd.read_csv(path + '/test.csv')

        leaderboard = api.competition_view_leaderboard(df) # Download the Kaggle competition leaderboard
        leader = get_leader(leaderboard) # Get the Leader
            
        res_kaggle.run_benchmark((train, test), dataframes[df]['task'], df, {'measure': dataframes[df]['measure'], 'score': leader['score']}, options)
    return res_kaggle.print_res()