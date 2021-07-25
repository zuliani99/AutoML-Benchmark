# Import needed
import os
import pandas as pd
from termcolor import colored
from sklearn.datasets import fetch_openml
import openml

# Function to get the target column
def get_target(train, test):
    for c in train.columns:
        if c not in test.columns:
            return c

def get_list_single_df(df):
    return [df] if isinstance(df, pd.DataFrame) else df


# Function for separating the target column from the rest of the DataFrame
def return_X_y(df):
    if not isinstance(df, tuple):
        # OpenML case
        return return_X_y_openML(df)
    target = get_target(df[0], df[1])
    # ATTENTION I USE ONLY THE TRAIN
    y = df[0][target]
    X = df[0].drop([target], axis=1)
    return X, y

# Kaggle case
def return_X_y_openML(df):
    new = df[0]
    y = new.iloc[:, -1].to_frame()
    X = new.iloc[:, :-1]
    return X, y

# Initial cleaning function of the DataFrame
def fill_and_to_category(dfs):
    dfs = get_list_single_df(dfs)
    for df in dfs:
        for col in df.columns:
            t = pd.api.types.infer_dtype(df[col])
            if t == "string":
                df[col] = pd.Categorical(df[col])
                df[col] = df[col].astype('category')
                df[col] = df[col].cat.add_categories('Unknown')
                df[col].fillna('Unknown', inplace =True)
                df[col] = df[col].cat.codes
            if t in ["integer", "floating"]:
                df[col] = df[col].fillna(df[col].mean())
            if t == 'categorical' :
                df[col] = df[col].cat.codes
    return dfs


# Function responsible for Downloading the DataFrame in the case of an OpenML Benchmark
def get_df_list(datalist, n_df, task):
    list_df = []
    for index, row in datalist.iterrows():
        file_dir = './dataframes/OpenML/'+ task +'/'
        name = str(row['did']) + '_' + str(row['name']) + '.csv'
        try:
            if not os.path.exists('./dataframes/OpenML/'+ task +'/' + name):
                X, y = fetch_openml(data_id=row['did'], as_frame=True, return_X_y=True, cache=True)
                if y is not None:
                    if not isinstance(y, pd.DataFrame):
                        y = y.to_frame()

                    X[y.columns[0]] = y

                    if n_df > 0:
                        print('------------------Dataset : ' + name + '------------------')

                        print(y.info())
                        fullname = os.path.join(file_dir, name)

                        print("good df " + fullname + '\n')

                        X[y.columns[0]] = y
                        X.to_csv(fullname, index=False, header=True)

                        list_df.append(fullname)

                        n_df-=1

            elif n_df > 0:
                print('------------------Dataset: ' + name + '------------------')
                print('-------------------------Dataset già presente-------------------------\n')
                list_df.append(file_dir + name)
                n_df-=1

        except Exception as e:
            print(colored("Can't download the DataFrame " + name + ' reason: '+ str(e)+ '\n','red'))

        if n_df == 0:
            break

    return list_df



# Function to search for a specific DataFrame in the previous results folders
def serch_df(df_id):
    for task in ['classification', 'regression']:
        lis = os.listdir('./dataframes/OpenML/'+ task +'/')
        for d in lis:
            if d.split('_')[0] == df_id:
                return './dataframes/OpenML/'+ task +'/' + d
    return None


# Function responsible for downloading the DataFrame coming from the ID list entered by the user
def download_dfs(ids):
    list_df = { 'classification': [], 'regression': []}
    for id in ids:
        search = serch_df(id) # Initially I check if the DataFrame that the user has chosen is present or not in one of the two folders
        try:
            task = None
            if search is None:
                # If there is no download through the appropriate API
                X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
                name = str(id)+ '_' +openml.datasets.get_dataset(id).name + '.csv'

                # Get the type of tasks
                tasks = openml.tasks.list_tasks(data_id=id, output_format="dataframe")
                ts = tasks['task_type'].unique()
                if ('Supervised Classification' not in ts and 'Supervised Regression' not in ts):
                    return 'Error: Invalid Dataframe task'
                task = 'classification' if 'Supervised Classification' in ts else 'regression'
                file_dir =  './dataframes/OpenML/' + task + '/'
                fullname = os.path.join(file_dir, name)


                if not isinstance(y, pd.DataFrame):
                    y = y.to_frame()
                X[y.columns[0]] = y

                df = X

                print(df.info())
                print(df.head())
                
                # Save the DataFrame in the corresponding folder
                df.to_csv(fullname, index=False, header=True)

            else:
                # If it is already present I save it and get the task
                fullname = search
                task = search.split('/')[3]
                print(pd.read_csv(fullname).head())

            list_df[task].append(fullname) # Add the dataframe path to its list_df dictionary array

        except Exception as e:
            # In case of error, I return a message
            return "Error: Can't download the DataFrame " + str(id) + ' reason: '+ str(e)
    list_df['classification'].extend(list_df['regression']) # Concatenation of the two arrays
    return list_df['classification']