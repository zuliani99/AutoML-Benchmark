from openml.tasks import TaskType
from utils.usefull_functions import get_df_list
import pandas as pd
import openml
import numpy as np
from utils.result_class import Result
from termcolor import colored

def openml_benchmark(df_n, morethan, options):
    print(df_n, morethan)
    list_df = []
    res_openml = Result('OpenML')

    tasks_class = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION), orient="index")
    tasks_reg = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_REGRESSION), orient="index")

    filtered_tasks_class = tasks_class.query("NumberOfInstances > " + str(morethan) + " and NumberOfInstances < " + str(2*morethan))
    filtered_tasks_reg = tasks_reg.query("NumberOfInstances > " + str(morethan) + " and NumberOfInstances < " + str(2*morethan))
    
    datalist_class = filtered_tasks_class[["did"]]
    datalist_reg = filtered_tasks_reg[["did"]]

    print(colored('--------------------------------Inizio Download Dataset --------------------------------', 'yellow'))

    list_df = get_df_list(datalist_class['did'].unique(), df_n, 'classification')
    list_df.extend(get_df_list(datalist_reg['did'].unique(), df_n, 'regression'))

    print(list_df)

    print(colored('--------------------------------Fine Dataset Download --------------------------------', 'yellow'))

    #ESECUZUONE DEGLI ALGORITMI
    for d in list_df:
        str_path = d.split('/')
        df = pd.read_csv(d)
        

        print('---------------------------------Dataset: ' + d + '---------------------------------')
        res_openml.run_benchmark(df, str_path[3], str_path[4], None, options)
        print(colored('--------------------------------- Riga inserita ---------------------------------', 'green'))

    return res_openml.print_res()