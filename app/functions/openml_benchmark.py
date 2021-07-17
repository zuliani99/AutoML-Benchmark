# Import necessari
from posixpath import lexists
from openml.tasks import TaskType
from utils.usefull_functions import get_df_list, download_dfs
import pandas as pd
import openml
from utils.result_class import Result
from termcolor import colored

# Funzione addetta all'esecuzione del OpenML Benchmark
def openml_benchmark(options, algo_options):
    print('openml_benchmark', options)
    res_openml = Result('OpenML') # Creazione di un nuovo Result di tipo OpenML

    if (isinstance(options, tuple)): # Se options è di tipo tupla allora avvio il benchmark con dei dataframe filtrati da openml tramite le opzioni inseirte
        list_df = openml_benchmark_unknowID(options)
    else:
        list = options.split(',')
        list_df = download_dfs([df for df in list]) # Altrimenti avvio il benchmark per openml con la lista dei DataFrame inserita dall'utente

    if isinstance(list_df, str):
        return list_df # Se il valore di ritorno è una lista vuol dire che c'è staato un errore durante il download di uno dei DataFrame

    # Esegui gli algoritmi per i DataFrame presenti in list_df
    for d in list_df:
        str_path = d.split('/')
        df = pd.read_csv(d)

        print('---------------------------------Dataset: ' + d + '---------------------------------')
        res_openml.run_benchmark(df, str_path[3], str_path[4], None, algo_options)
        print(colored('--------------------------------- Riga inserita ---------------------------------', 'green'))

    return res_openml.print_res()

def openml_benchmark_unknowID(options):
    df_n, morethan = options
    # Ottemgo tutti i task di tipo SUPERVISED_CLASSIFICATION e SUPERVISED_REGRESSION tramite l'API di OpenML
    tasks_class = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION), orient="index")
    tasks_reg = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_REGRESSION), orient="index")

    # Filtro entrambi i DataFrame per il numero di istanze minime e massime
    filtered_tasks_class = tasks_class.query("NumberOfInstances > " + str(morethan) + " and NumberOfInstances < " + str(2*morethan))
    filtered_tasks_reg = tasks_reg.query("NumberOfInstances > " + str(morethan) + " and NumberOfInstances < " + str(2*morethan))

    # Elimino i duplicati e tutte le colonne eccetto did e name
    datalist_class = filtered_tasks_class[["did", "name"]].drop_duplicates(subset=['did' and 'name'], inplace=False)
    datalist_reg = filtered_tasks_reg[["did", "name"]].drop_duplicates(subset=['did' and 'name'], inplace=False)

    print(colored('--------------------------------Inizio Download Dataset --------------------------------', 'yellow'))

        # Creo un unico DataFrame dato dal download di df_n Dataframe per il caso di calsssificazione e di di df_n Dataframe per regressione
    result = get_df_list(datalist_class, df_n, 'classification')
    result.extend(get_df_list(datalist_reg, df_n, 'regression'))

    print(result)

    print(colored('--------------------------------Fine Dataset Download --------------------------------', 'yellow'))
    return result