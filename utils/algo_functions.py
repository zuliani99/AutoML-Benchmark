from algorithms.auto_sklearn import auto_sklearn
from algorithms.tpot import TPOT
from algorithms.h2o import H2O
from algorithms.auto_keras import autokeras
from algorithms.auto_gluon import autogluon

def fun_autosklearn(df, task):
    res_autosklearn = 0.0
    print("--------------------------------AUTOSKLEARN--------------------------------")
    try:
        res_autosklearn = (auto_sklearn(df, task))
        print('Risultato memorizzato!')
        return res_autosklearn
    except:
        print('Qualcosa è andato storto :(')
    print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
    return res_autosklearn


def fun_tpot(df, task):
    res_tpot = 0.0
    print("-----------------------------------TPOT------------------------------------")
    try:
        res_tpot = (TPOT(df, task))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("-----------------------------------TPOT------------------------------------\n\n")
    return res_tpot


def fun_autokeras(df, task):
    res_autokeras = 0.0
    print("---------------------------------AUTOKERAS---------------------------------")
    try:
        res_autokeras = (autokeras(df, task))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("---------------------------------AUTOKERAS---------------------------------\n\n")
    return res_autokeras

def fun_h2o(df, task):
    res_h2o = 0.0
    print("------------------------------------H2O------------------------------------")
    try:
        res_h2o = (H2O(df, task))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("------------------------------------H2O------------------------------------\n\n")
    return res_h2o


def fun_autogluon(df, task):
    res_autogluon = 0.0
    print("----------------------------------AUTOGLUON--------------------------------")
    try:
        res_autogluon = (autogluon(df, task))
        print('Risultato memorizzato!')
    except:
        print('Qualcosa è andato storto :(')
    print("----------------------------------AUTOGLUON--------------------------------\n\n")
    return res_autogluon