# Import needed
import pandas as pd
import os
from algorithms.auto_sklearn import auto_sklearn
from algorithms.auto_gluon import autogluon
from algorithms.tpot import TPOT
from algorithms.h2o import H2O
from algorithms.mljar_supervised import mljar
from termcolor import colored
from utils.usefull_functions import download_dfs
import time

# Dictionary used as a switch case for running one or all algorithms for a given DataFrame
def switch(algo, df, name, task, options):
    return {
        'autosklearn': lambda df, task: auto_sklearn(df, task, options['autosklearn'], time.time()),
        'tpot': lambda df, task: TPOT(df, task, options['tpot'], time.time()),
        'h2o': lambda df, task: H2O(df, task, options['h2o'], time.time()),
        'mljar': lambda df, task: mljar(df, task, options['mljar'], time.time()),
        'autogluon': lambda df, task: autogluon(df, task, options['autogluon'], time.time()),
        'all': lambda df, task: pd.DataFrame.from_dict({
                                                        'dataframe': name,
                                                        'autosklearn': auto_sklearn(df, task, options['autosklearn'], time.time()),
                                                        'tpot': TPOT(df, task, options['tpot'], time.time()),
                                                        'h2o': H2O(df, task, options['h2o'], time.time()),
                                                        'mljar': mljar(df, task, options['mljar'], time.time()),
                                                        'autogluon': autogluon(df, task, options['autogluon'], time.time())})
    }.get(algo)(df, task)

def test(id, algo, options):
    print('----------------'+str(id)+'-----------'+str(algo)+'-------------')
    # Download the dataframe
    df = download_dfs([id])
    if isinstance(df, str):
        # Return an exception on an error
        return None, df
    try:
        # Running the test
        res = switch(algo, pd.read_csv(df[0]), df[0].split('/')[4], df[0].split('/')[3], options)
        return df[0].split('/')[3], res
    except Exception as e:
        # Return an exception on an error
        text = 'An error occured during the benchmak of the dataframe: ' + str(id) + ' reason: ' + str(e)
        print(colored(text + '\n', 'red'))
        return None, text