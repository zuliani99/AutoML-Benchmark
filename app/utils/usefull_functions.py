import os
import pandas as pd
from termcolor import colored
from sklearn.datasets import fetch_openml

def get_target(train, test):
    for c in train.columns:
        if c not in test.columns:
            return c

def get_list_single_df(df):
    return [df] if isinstance(df, pd.DataFrame) else df


def return_X_y(df):
    if not isinstance(df, tuple):
        return return_X_y_openML(df)
    target = get_target(df[0], df[1])
    y = df[0][target]
    X = df[0].drop([target], axis=1)
    return X, y, None

def return_X_y_openML(df):
    new = df[0]
    n_target = new['n_target'][0]
    new = new.drop('n_target', axis = 1)

    y = new.iloc[:, -n_target].to_frame()
    X = new.iloc[:, :-n_target]
    return X, y, n_target

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



def get_df_list(datalist, n_df, task):
    list_df = []
    for row in datalist:
        try:

            file_dir = './datasets/OpenML/'+ task +'/'

            if not os.path.exists('./datasets/OpenML/'+ task +'/' + str(row) + '.csv'):
                X, y = fetch_openml(data_id=row, as_frame=True, return_X_y=True, cache=True)
                if y is not None:
                    if not isinstance(y, pd.DataFrame):
                        y = y.to_frame()

                    if (len(y.columns) == 1):
                        X[y.columns[0]] = y
                    else:
                        for col in y.columns:
                            X[col] = y[col]
                    df = X
                    df['n_target'] = len(y.columns)



                    if n_df > 0:
                        print('------------------Dataset ID: ' + str(row) + '------------------')

                        print(y.info())
                        fullname = os.path.join(file_dir, str(row) + '.csv')

                        print("good df " + fullname + '\n')

                        X[y.columns[0]] = y
                        X.to_csv(fullname, index=False, header=True)

                        list_df.append(fullname)

                        n_df-=1

            elif n_df > 0:
                print('------------------Dataset ID: ' + str(row) + '------------------')
                print('-------------------------Dataset già presente-------------------------\n')
                list_df.append(file_dir + str(row) + '.csv')
                n_df-=1

        except Exception as e:
            print(colored('Impossibile scaricare il DataFrame '+ str(row)+ ' causa: '+ str(e)+ '\n','red'))

        if n_df == 0:
            break

    return list_df