from sklearn.datasets import fetch_openml
import os
import pandas as pd
import openml
from utils.result_class import Result

def openml_benchmark(df_n, morethan):
    n_class = df_n
    n_reg = df_n
    list_df = []

    res_openml = Result('OpenML')

    openml_list = openml.datasets.list_datasets()  # returns a dict
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist = datalist[["did", "name", "NumberOfInstances"]]
    df_id_name = datalist[datalist.NumberOfInstances > morethan].sort_values(["NumberOfInstances"]) #.head(df_n)

    print('--------------------------------Inizio Dataset Download--------------------------------')

    # DOWNLOA DEI DATASETS
    for index, row in df_id_name.iterrows():
        try:
            if not os.path.exists('./datasets/classification/' + str(row['did']) + '.csv') and not os.path.exists('./datasets/regression/' + str(row['did']) + '.csv'):
                X, y = fetch_openml(data_id=row[0], as_frame=True, return_X_y=True, cache=True)
                if y is not None and not isinstance(y, pd.DataFrame):
                    y = y.to_frame()
                else:
                    y = X.iloc[:, -1].to_frame()
                    X = X.drop(y.columns[0], axis=1)

                t = pd.api.types.infer_dtype(y[y.columns[0]])
                if (t == "categorical" or t == "boolean"):
                    file_dir =  './datasets/classification/'

                if (t == "floating" or t == 'integer' or t == 'decimal'):
                    file_dir =  './datasets/regression/'

                if n_class > 0 or n_reg > 0:
                    print('------------------Dataset ID: ' + str(row['did']) + ' name: ' + str(row['name']) + '------------------')
                    print(y.info())
                    fullname = os.path.join(file_dir, str(row['did']) + '.csv')

                    print("good df " + fullname + '\n')

                    X[y.columns[0]] = y
                    X.to_csv(fullname, index=False, header=True)

                    list_df.append(fullname)

                    if file_dir == './datasets/classification/':
                        n_class-=1
                    else:
                        n_reg-=1
            else:
                if os.path.exists('./datasets/classification/' + str(row['did']) + '.csv') and n_class > 0:
                    print('------------------Dataset ID: ' + str(row['did']) + ' name: ' + str(row['name']) + '------------------')
                    print('-------------------------Dataset già presente-------------------------\n')
                    list_df.append('./datasets/classification/' + str(row['did']) + '.csv')
                    n_class-=1
                if os.path.exists('./datasets/regression/' + str(row['did']) + '.csv') and n_reg > 0:
                    print('------------------Dataset ID: ' + str(row['did']) + ' name: ' + str(row['name']) + '------------------')
                    print('-------------------------Dataset già presente-------------------------\n')
                    list_df.append('./datasets/regression/' + str(row['did']) + '.csv')
                    n_reg-=1

        except:
            print("Impossibile scaricare il DataFrame\n")

        if n_class == 0 and n_reg == 0:
            print('--------------------------------Fine Dataset Download--------------------------------')
            break
    

    #ESECUZUONE DEGLI ALGORITMI
    for d in list_df:
        task = d.split('/')[2]
        df = pd.read_csv(d)
                
        print('---------------------------------Dataset: ' + d + '---------------------------------\n')
        res_openml.run_benchmark(df, task)

    res_openml.print_res()