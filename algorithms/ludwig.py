from ludwig.api import LudwigModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import logging
from ludwig.utils.data_utils import load_json
import os
import numpy as np



def create_dict(label, target):
    model = { 'input_features': [], 'output_features': [], 'training':{}}
    
    for lab in label.columns:
        #print(lab + " " +pd.api.types.infer_dtype(label[lab]))
        t = pd.api.types.infer_dtype(label[lab])
    
        if t == "floating" or t == "integer":
            model['input_features'].append({'name': lab, 'type': 'numerical'})
        if t == "categorical":
            model['input_features'].append({'name': lab, 'type': 'category'})
        if t == "boolean":
            model['input_features'].append({'name': lab, 'type': 'binary'})
        if t == "string":
            model['input_features'].append({'name': lab, 'type': 'text'})
    
    t = pd.api.types.infer_dtype(target[target.columns[0]])
    if t == "floating" or t == "integer":
        model['output_features'].append({ 'name': target.columns[0], 'type': 'numerical' })
    if t == "categorical":
        model['output_features'].append({ 'name': target.columns[0], 'type': 'category' })
    if t == "string":
        model['output_features'].append({ 'name': target.columns[0], 'type': 'text' })
    
    #, 'epochs': 10
    model['training'] = ({'validation_field': target.columns[0], 'validation_metric': 'last_accuracy', 'epochs': 5})
    print(model)
    return model



def delete_folder():
    dir_path = './results/api_experiment_run'
    if os.path.exists(dir_path):
        try:
            os.rmdir(dir_path)
            print("Delete ------------------------------> DONE")
        except OSError as e:
            print("Error:                   %s : %s" % (dir_path, e.strerror))




def get_results(target):
    experiment_model_dir = './results/api_experiment_run'
    train_stats = load_json(os.path.join(experiment_model_dir,'training_statistics.json'))
    print(train_stats)
    
    index = np.argmax(train_stats['validation'][target]['last_accuracy'])
    validation = train_stats['validation'][target]['last_accuracy'][index]
    test = train_stats['test'][target]['last_accuracy'][index]

    return (index+1, validation, test)




def ludwig_class(df):
    delete_folder()

    
    for col in df.columns:
        t = pd.api.types.infer_dtype(df[col])
        if t == "categorical":
            df[col] = df[col].astype('object')

    y = df.iloc[:, -1].to_frame()
    X = df.iloc[:, :-1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    X_train[y_train.columns[0]] = y_train
    #train = X_train.convert_dtypes()


    print(X_train.dtypes)
    print("-----------------------------------")
    print(X_train.head())

    print("----------------------------------------------------------------------")

    print(X_test.info())
    print("-----------------------------------")
    print(X_test.head())
    
    model = LudwigModel(create_dict(X, y), logging_level=logging.INFO)
    
    train_stats = model.train(dataset=X_train, logging_level=logging.INFO)

    test_stats = model.predict(X_test)



    return get_results(y.columns[0])