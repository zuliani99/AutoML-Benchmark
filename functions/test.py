from sklearn.datasets import fetch_openml
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon
import pandas as pd
import os
import numpy as np
from algorithms.auto_keras import autokeras
from algorithms.auto_sklearn import auto_sklearn
from algorithms.auto_gluon import autogluon
from algorithms.tpot import TPOT
from algorithms.h2o import H2O
from termcolor import colored
import openml

def switch(algo, df, task):
    return {
        'autosklearn': lambda df, task: auto_sklearn(df, task),
        'tpot': lambda df, task: TPOT(df, task),
        'h2o': lambda df, task: H2O(df, task),
        'autokeras': lambda df, task: autokeras(df, task),
        'autogluon': lambda df, task: autogluon(df, task),
        'all': lambda df, task: pd.DataFrame.from_dict({'autosklearn': auto_sklearn(df, task),
                                                        'tpot': TPOT(df, task),
                                                        'autokeras': autokeras(df, task),
                                                        'h2o': H2O(df, task),
                                                        'autogluon': autogluon(df, task)})
    }.get(algo)(df, task)

def test(id, algo):
    print('----------------'+str(id)+'-----------'+str(algo)+'-------------')
    try:
        if not os.path.exists('./datasets/classification/' + str(id) + '.csv') and not os.path.exists('./datasets/regression/' + str(id) + '.csv'):
            X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
            if not isinstance(y, pd.DataFrame):
                y = y.to_frame()
            if(len(y.columns) == 1):
                X[y.columns[0]] = y
                df = X
            else:
                for col in y.columns:
                    X[col] = y[col]
                df = X

            df['n_target'] = len(y.columns)

            print(df.info())
            print(df.head())

            tasks = openml.tasks.list_tasks(data_id=id, output_format="dataframe")
            ts = tasks['task_type'].unique()
            if 'Supervised Classification' in ts or 'Supervised Regression' in ts:
                if 'Supervised Classification' in ts:
                    task = 'classification'
                elif 'Supervised Regression' in ts:
                    task = 'regression'

                file_dir =  './datasets/' + task + '/'
                fullname = os.path.join(file_dir, str(id) + '.csv')
                df.to_csv(fullname, index=False, header=True)
                res = switch(algo, df, task)
                print(task, res)
                return task, res
            else:
                return None, None            
        else:
            if os.path.exists('./datasets/classification/' + str(id) + '.csv'):
                task = 'classification'
                path = './datasets/classification/' + str(id) + '.csv'
            else:
                task = 'regression'
                path = './datasets/regression/' + str(id) + '.csv'
            
            df = pd.read_csv(path)

            print(df.head())
            res = switch(algo, df, task)
            print(task, res)
            # ritorno il tipo di task ed il risultato dell'algoritmo -> (acc, f1) o (rmse, r2) oppure il datafrsame con tutti i risultati di ytutti gli algoritmi
            return task, res
    except Exception as e:
        text = 'Impossibile scaricare il DataFrame ' + str(id) + ' causa: ' + str(e)
        print(colored(text + '\n', 'red'))
        return None, text