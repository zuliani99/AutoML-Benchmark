import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from utils.result_class import Result

datasets = [('titanic', 'classification'), ('tabular-playground-series-mar-2021', 'classification')]

def kaggle_benchmark(list_df):
    api = KaggleApi()
    api.authenticate()

    res_kaggle = Result('Kaggle')

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

            res_kaggle.run_benchmark((train, test), task)
        else:
            print('\nDatasek di kaggle inesistente. Se esistente accertarsi di aver accettato le condizioni della competizione.\n')

    res_kaggle.print_res()