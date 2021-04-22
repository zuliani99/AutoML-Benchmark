from sklearn.datasets import fetch_openml
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon
import pandas as pd
import os
from utils.algo_functions import auto_sklearn, H2O, TPOT, autokeras, autogluon

def switch(algo, df, task):
    return {
        'autosklearn': lambda df, task: auto_sklearn(df, task),
        'tpot': lambda df, task: TPOT(df, task),
        'h2o': lambda df, task: H2O(df, task),
        'autokeras': lambda df, task: autokeras(df, task),
        'autogluon': lambda df, task: autogluon(df, task),
        'all': lambda df, task: pd.DataFrame.from_dict({'autosklearn': fun_autosklearn(df, task),
                                                        'tpot': fun_tpot(df, task),
                                                        'autokeras': fun_autokeras(df, task),
                                                        'h2o': fun_h2o(df, task),
                                                        'autogluon': fun_autogluon(df, task)})
    }.get(algo)(df, task)

def test(id, algo):
    if not os.path.exists('./datasets/classification/' + str(id) + '.csv') and not os.path.exists('./datasets/regression/' + str(id) + '.csv'):
        X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
        y = y.to_frame()
        X[y.columns[0]] = y
        df = X
        t = pd.api.types.infer_dtype(y[y.columns[0]])
        if (t == "categorical" or t == "boolean"):
            file_dir =  './datasets/classification/'

        if (t == "floating" or t == 'integer' or t == 'decimal'):
            file_dir =  './datasets/regression/'

        fullname = os.path.join(file_dir, str(id) + '.csv')
        df.to_csv(fullname, index=False, header=True)
    else:
        if os.path.exists('./datasets/classification/' + str(id) + '.csv'):
            task = 'classification'
            path = './datasets/classification/' + str(id) + '.csv'
        else:
            task = 'regression'
            path = './datasets/regression/' + str(id) + '.csv'
        
        df = pd.read_csv(path)

        print(switch(algo, df, task))
