# Import needed
from openml.tasks import TaskType
from utils.usefull_functions import get_df_list, download_dfs
import pandas as pd
import openml
from utils.result_class import Result
from termcolor import colored

# Function responsible for executing the OpenML Benchmark
def openml_benchmark(options, algo_options):
    print("--------------------------------- Starting OpenML Benchmark ---------------------------------")
    res_openml = Result('OpenML') # Creation of a new Result of type OpenML

    if (isinstance(options, tuple)): # If options is of type tuple then I start the benchmark with dataframes filtered by openml through the options inseirte
        list_df = openml_benchmark_unknowID(options)
    else:
        list = options.split(',')
        list_df = download_dfs([df for df in list]) # Otherwise I start the benchmark for OpenML with the list of DataFrames entered by the user

    if isinstance(list_df, str):
        return list_df # If the return value is a list it means that there was an error while downloading one of the DataFrame

    # Execute the algorithms for the DataFrame present in list_df
    for d in list_df:
        str_path = d.split('/')
        df = pd.read_csv(d)

        print('--------------------------------- Dataframe: ' + d + ' ---------------------------------')
        res_openml.run_benchmark(df, str_path[3], str_path[4], None, algo_options)
        print(colored('--------------------------------- Row Inserted ---------------------------------', 'green'))

    return res_openml.print_res()

def openml_benchmark_unknowID(options):
    df_n, morethan = options
    # I accomplish all SUPERVISED_CLASSIFICATION and SUPERVISED_REGRESSION type tasks through the OpenML API
    tasks_class = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_CLASSIFICATION), orient="index")
    tasks_reg = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_type=TaskType.SUPERVISED_REGRESSION), orient="index")

    # Filter both DataFrames by the number of minimum and maximum instances
    filtered_tasks_class = tasks_class.query("NumberOfInstances > " + str(morethan))
    filtered_tasks_reg = tasks_reg.query("NumberOfInstances > " + str(morethan))

    # Eliminate duplicates and all columns except did and name
    datalist_class = filtered_tasks_class[["did", "name"]].drop_duplicates(subset=['did' and 'name'], inplace=False)
    datalist_reg = filtered_tasks_reg[["did", "name"]].drop_duplicates(subset=['did' and 'name'], inplace=False)

    print(colored('-------------------------------- Starting Dataframes Download --------------------------------', 'yellow'))

    # Create a single DataFrame given by the download of df_n Dataframe for the classification case and df_n Dataframe for the regression
    result = get_df_list(datalist_class, df_n, 'classification')
    result.extend(get_df_list(datalist_reg, df_n, 'regression'))

    print(result)

    print(colored('-------------------------------- Ended Dataframes Download --------------------------------', 'yellow'))
    return result