# Import necessari
from openml.tasks import TaskType
from utils.usefull_functions import get_df_list
import pandas as pd
import openml
from utils.result_class import Result
from termcolor import colored

# Funzione addetta all'esecuzione del OpenML Benchmark
def openml_benchmark(df_n, morethan, options):
    print(df_n, morethan)
    list_df = []
    res_openml = Result('OpenML') # Creazione di un nuovo Result con tipo OpenML

    # Ottemgo tutti i task di tipo SUPERVISED_CLASSIFICATION e SUPERVISED_REGRESSION tramite l'API di OpenML
    tasks_class = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION), orient="index")
    tasks_reg = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_REGRESSION), orient="index")

    # Filtro entrambi i DataFrame per il numero di istanze minime e massime
    filtered_tasks_class = tasks_class.query("NumberOfInstances > " + str(morethan) + " and NumberOfInstances < " + str(2*morethan))
    filtered_tasks_reg = tasks_reg.query("NumberOfInstances > " + str(morethan) + " and NumberOfInstances < " + str(2*morethan))
    
    # Elimino i duooplicati e tutte le colonne eccetto did e name
    datalist_class = filtered_tasks_class[["did", "name"]].drop_duplicates(subset=['did' and 'name'], inplace=False)
    datalist_reg = filtered_tasks_reg[["did", "name"]].drop_duplicates(subset=['did' and 'name'], inplace=False)

    print(colored('--------------------------------Inizio Download Dataset --------------------------------', 'yellow'))

    # Creo un unico DataFrame dato dal download di df_n Dataframe per il caso di calsssificazione e di di df_n Dataframe per regressione
    list_df = get_df_list(datalist_class, df_n, 'classification')
    list_df.extend(get_df_list(datalist_reg, df_n, 'regression'))

    print(list_df)

    print(colored('--------------------------------Fine Dataset Download --------------------------------', 'yellow'))

    # Esegui gli algoritmi per i DataFrame presenti in list_df
    for d in list_df:
        str_path = d.split('/')
        df = pd.read_csv(d)

        print('---------------------------------Dataset: ' + d + '---------------------------------')
        res_openml.run_benchmark(df, str_path[3], str_path[4], None, options)
        print(colored('--------------------------------- Riga inserita ---------------------------------', 'green'))

    return res_openml.print_res()