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
        if not os.path.exists('./dataframes/OpenML/classification/' + str(id) + '.csv') and not os.path.exists('./dataframes/OpenML/regression/' + str(id) + '.csv'):
            X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
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
            fullname = os.path.join(file_dir, str(id) + '.csv')
            df.to_csv(fullname, index=False, header=True)
        else:
            if os.path.exists('./dataframes/OpenML/classification/' + str(id) + '.csv'):
                task = 'classification'
                path = './dataframes/OpenML/classification/' + str(id) + '.csv'
            else:
                task = 'regression'
                path = './dataframes/OpenML/regression/' + str(id) + '.csv'

            df = pd.read_csv(path)

            print(df.head())
        res = switch(algo, df, task, options)
        print(task, res)
        return task, res
    except Exception as e:
        text = "An error occured during the benchmak of the dataframe: " + str(id) + ' reason: ' + str(e)
        print(colored(text + '\n', 'red'))
        return None, text