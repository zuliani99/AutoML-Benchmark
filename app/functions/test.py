# Import necessari
from sklearn.datasets import fetch_openml
import pandas as pd
import os
from algorithms.auto_keras import autokeras
from algorithms.auto_sklearn import auto_sklearn
from algorithms.auto_gluon import autogluon
from algorithms.tpot import TPOT
from algorithms.h2o import H2O
from termcolor import colored
from utils.usefull_functions import download_dfs



# Dizionario utilizzato come swith case per l'esecizione di uno o tutti gli algoritmi per un determinato DataFrame
def switch(algo, df, name, task, options):
    return {
        'autosklearn': lambda df, task: auto_sklearn(df, task, options['autosklearn']),
        'tpot': lambda df, task: TPOT(df, task, options['tpot']),
        'h2o': lambda df, task: H2O(df, task, options['h2o']),
        'autokeras': lambda df, task: autokeras(df, task, options['autokeras']),
        'autogluon': lambda df, task: autogluon(df, task, options['autogluon']),
        'all': lambda df, task: pd.DataFrame.from_dict({
                                                        'dataframe': name,
                                                        'autosklearn': auto_sklearn(df, task, options['autosklearn']),
                                                        'tpot': TPOT(df, task, options['tpot']),
                                                        'h2o': H2O(df, task, options['autokeras']),
                                                        'autokeras': autokeras(df, task, options['h2o']),
                                                        'autogluon': autogluon(df, task, options['autogluon'])})
    }.get(algo)(df, task)

def test(id, algo, options):
    print('----------------'+str(id)+'-----------'+str(algo)+'-------------')
    try:
        # Esecuzione del test
        df =  download_dfs((id))
        res = switch(algo, pd.read_csv(df[0]), df[0].split('/')[4], df[0].split('/')[3], options)
        return df[0].split('/')[3], res
    except Exception as e:
        # Ritorno di un eccezione in caso di errore
        text = 'An error occured during the benchmak of the dataframe: ' + str(id) + ' reason: ' + str(e)
        print(colored(text + '\n', 'red'))
        return None, text