from sklearn.datasets import fetch_openml
#from openml.datasets import get_dataset
#import openml
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon
import pandas as pd
import os
from algorithms.auto_keras import autokeras
from algorithms.auto_sklearn import auto_sklearn
from algorithms.auto_gluon import autogluon
from algorithms.tpot import TPOT
from algorithms.h2o import H2O
from termcolor import colored

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
    if not os.path.exists('./datasets/classification/' + str(id) + '.csv') and not os.path.exists('./datasets/regression/' + str(id) + '.csv'):
        X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
        #dataset = openml.datasets.get_dataset(row[0])
        #X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        if y is None:
            print(colored('Dataset senza una target feature', 'red'))
        else:
            if not isinstance(y, pd.DataFrame):
                y = y.to_frame()
            #else:
                #y = X.iloc[:, -1].to_frame()
                #X = X.drop(y.columns[0], axis=1)

            print(y.info())
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
            t = pd.api.types.infer_dtype(y[y.columns[0]])
            if (t == "categorical" or t == "boolean"):
                task = 'classification'

            if (t == "floating" or t == 'integer' or t == 'decimal'):
                task = 'regression'

            file_dir =  './datasets/' + task + '/'
            fullname = os.path.join(file_dir, str(id) + '.csv')
            df.to_csv(fullname, index=False, header=True)
            
            print(switch(algo, df, task))
    else:
        if os.path.exists('./datasets/classification/' + str(id) + '.csv'):
            task = 'classification'
            path = './datasets/classification/' + str(id) + '.csv'
        else:
            task = 'regression'
            path = './datasets/regression/' + str(id) + '.csv'
        
        df = pd.read_csv(path)

        print(df.head())

        print(switch(algo, df, task))
