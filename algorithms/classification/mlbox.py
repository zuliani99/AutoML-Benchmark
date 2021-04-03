#pip3 install mlbox

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *

def prepare_and_test():
    paths = ["../train.csv", "../test.csv"]
    #print(paths)
    train = pd.read_csv("../train.csv")
    target_name = train.columns[train.shape[1]-1]
    data = Reader(sep=",").train_test_split(paths, target_name)  #reading
    data = Drift_thresholder().fit_transform(data)  #deleting non-stable variables
    opt = Optimiser(scoring = 'accuracy', n_folds = 15)
    opt.evaluate(None, data)
    space = {
        
            'ne__numerical_strategy':{"search":"choice",
                                    "space":[0]},
            'ce__strategy':{"search":"choice",
                            "space":["label_encoding","random_projection", "entity_embedding"]}, 
            'fs__threshold':{"search":"uniform",
                            "space":[0.01,0.3]},    
            'est__max_depth':{"search":"choice",
                                    "space":[3,4,5,6,7]}
        
            }

    best = opt.optimise(space, data, 15)
    prd = Predictor()
    return (prd.fit_predict(best, data))

def mlbx_class (df):
    y = df.iloc[:, -1:]
    X = df.iloc[:, 0:df.shape[1]-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    y_train = y_train.to_frame() 
    X_train[y_train.columns[0]] = y_train
    train = X_train
    test = X_test

    train.to_csv("../train.csv", index=False, header=True)
    test.to_csv("../test.csv", index=False, header=True)

    return(prepare_and_test())