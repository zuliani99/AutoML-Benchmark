from typing import List
from sklearn.datasets import fetch_openml
import pandas as pd
import os
from algorithms.auto_keras import autokeras
from algorithms.auto_sklearn import auto_sklearn
from algorithms.auto_gluon import autogluon
from algorithms.tpot import TPOT
from algorithms.h2o import H2O
from termcolor import colored
import openml

def serch_df(df_id):
    for task in ['classification', 'regression']:
        lis = os.listdir('./dataframes/OpenML/'+ task +'/')
        for d in lis:
            if d.split('_')[0] == df_id:
                return d
    return None

def switch(algo, df, task, options):
    return {
        'autosklearn': lambda df, task: auto_sklearn(df, task, int(options['autosklearn']['time'])),
        'tpot': lambda df, task: TPOT(df, task, int(options['tpot']['time'])),
        'h2o': lambda df, task: H2O(df, task, int(options['h2o']['time'])),
        'autokeras': lambda df, task: autokeras(df, task, int(options['autokeras']['time'])),
        'autogluon': lambda df, task: autogluon(df, task, int(options['autogluon']['time'])),
        'all': lambda df, task: pd.DataFrame.from_dict({'autosklearn': auto_sklearn(df, task, int(options['autosklearn']['time'])),
                                                        'tpot': TPOT(df, task, int(options['tpot']['time'])),
                                                        'autokeras': autokeras(df, task, int(options['h2o']['time'])),
                                                        'h2o': H2O(df, task, int(options['autokeras']['time'])),
                                                        'autogluon': autogluon(df, task, int(options['autogluon']['time']))})
    }.get(algo)(df, task)

def test(id, algo, options):
    print('----------------'+str(id)+'-----------'+str(algo)+'-------------')
    try:
        search = serch_df(id)
        if search is None:
            X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
            name = str(id)+ '_' +openml.datasets.get_dataset(id).name + '.csv'

            if not isinstance(y, pd.DataFrame):
                y = y.to_frame()
            X[y.columns[0]] = y

            df = X

            print(df.info())
            print(df.head())

            tasks = openml.tasks.list_tasks(data_id=id, output_format="dataframe")
            ts = tasks['task_type'].unique()
            if ('Supervised Classification' not in ts and 'Supervised Regression' not in ts):
                return None, None
            task = 'classification' if 'Supervised Classification' in ts else 'regression'
            file_dir =  './dataframes/OpenML/' + task + '/'
            fullname = os.path.join(file_dir, name)
            df.to_csv(fullname, index=False, header=True)
            
        else:
            df = pd.read_csv(search)
            task = search.split('/')[3]
            print(df.head())

        res = switch(algo, df, task, options)
        print(task, res)
        return task, res
    except Exception as e:
        text = "An error occured during the benchmak of the dataframe: " + str(id) + ' reason: ' + str(e)
        print(colored(text + '\n', 'red'))
        return None, text