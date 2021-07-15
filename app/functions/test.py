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
import openml

# Funzione per la ricerca dii un specifico DataFrame nelle cartelle dei risultati precedenti
def serch_df(df_id):
    for task in ['classification', 'regression']:
        lis = os.listdir('./dataframes/OpenML/'+ task +'/')
        for d in lis:
            if d.split('_')[0] == df_id:
                return d
    return None

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
        return do_test(id, algo, options)
    except Exception as e:
        # Ritorno di un eccezione in caso di errore
        text = 'An error occured during the benchmak of the dataframe: ' + str(id) + ' reason: ' + str(e)
        print(colored(text + '\n', 'red'))
        return None, text

def do_test(id, algo, options):
    search = serch_df(id) # Inizialmente controllo se il DataFrame che ha scelto l'utente sia presente o meno in una delle due cartelle
    if search is None:
        # Se non è presente lo scarico attraverso lapposita API
        X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
        name = str(id)+ '_' +openml.datasets.get_dataset(id).name + '.csv'

        if not isinstance(y, pd.DataFrame):
            y = y.to_frame()
        X[y.columns[0]] = y

        df = X

        print(df.info())
        print(df.head())
        
        # Ottengo il tipo di task
        tasks = openml.tasks.list_tasks(data_id=id, output_format="dataframe")
        ts = tasks['task_type'].unique()
        if ('Supervised Classification' not in ts and 'Supervised Regression' not in ts):
            return None, None
        task = 'classification' if 'Supervised Classification' in ts else 'regression'
        file_dir =  './dataframes/OpenML/' + task + '/'
        fullname = os.path.join(file_dir, name)

        # Effettuo il salvataggio del DataFrame nell cartella corrispondente
        df.to_csv(fullname, index=False, header=True)

    else:
        # Se è già presente lo leggo e ottengo il task
        df = pd.read_csv(search)
        task = search.split('/')[3]
        print(df.head())

    # Esegui il test benchmark
    res = switch(algo, df, name, task, options)
    return task, res