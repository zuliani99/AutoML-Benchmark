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
    # ATTENZIONE USO SOLO IL TRAIN
    y = df[0][target]
    X = df[0].drop([target], axis=1)
    return X, y

def return_X_y_openML(df):
    new = df[0]
    #n_target = new['n_target'][0]
    #new = new.drop('n_target', axis = 1)

    y = new.iloc[:, -1].to_frame()
    X = new.iloc[:, :-1]
    return X, y

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
    for index, row in datalist.iterrows():
        print('rigaaaaaaaa ', row['did'], row['name'])
        file_dir = './dataframes/OpenML/'+ task +'/'
        name = str(row['did']) + '_' + str(row['name']) + '.csv'
        try:
            if not os.path.exists('./dataframes/OpenML/'+ task +'/' + name):
                X, y = fetch_openml(data_id=row['did'], as_frame=True, return_X_y=True, cache=True)
                if y is not None:
                    if not isinstance(y, pd.DataFrame):
                        y = y.to_frame()

                    #if (len(y.columns) == 1):
                    X[y.columns[0]] = y
                    #else:
                        #for col in y.columns:
                            #X[col] = y[col]
                    df = X
                    #df['n_target'] = len(y.columns)



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