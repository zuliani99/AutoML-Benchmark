from algorithms.auto_sklearn import auto_sklearn
from algorithms.tpot import TPOT
from algorithms.h2o import H2O
from algorithms.auto_keras import autokeras
from algorithms.auto_gluon import autogluon
from termcolor import colored

def fun_autosklearn(df, task, timelife):
    res_autosklearn = (0.0, 0.0)
    print("--------------------------------AUTOSKLEARN--------------------------------")
    try:
        res_autosklearn = (auto_sklearn(df, task, timelife))
        print(colored('Risultato memorizzato!', 'green'))
        return res_autosklearn
    except Exception as e:
        print(colored('Qualcosa è andato storto :(     -> ' + str(e), 'red'))
    print("--------------------------------AUTOSKLEARN--------------------------------\n\n")
    return res_autosklearn


def fun_tpot(df, task, timelife):
    res_tpot = (0.0, 0.0)
    print("-----------------------------------TPOT------------------------------------")
    try:
        res_tpot = (TPOT(df, task, timelife))
        print(colored('Risultato memorizzato!', 'green'))
    except Exception as e:
        print(colored('Qualcosa è andato storto :(     -> ' + str(e), 'red'))
    print("-----------------------------------TPOT------------------------------------\n\n")
    return res_tpot


def fun_autokeras(df, task, timelife):
    res_autokeras = (0.0, 0.0)
    print("---------------------------------AUTOKERAS---------------------------------")
    try:
        res_autokeras = (autokeras(df, task, timelife))
        print(colored('Risultato memorizzato!', 'green'))
    except Exception as e:
        print(colored('Qualcosa è andato storto :(     -> ' + str(e), 'red'))
    print("---------------------------------AUTOKERAS---------------------------------\n\n")
    return res_autokeras

def fun_h2o(df, task, timelife):
    res_h2o = (0.0, 0.0)
    print("------------------------------------H2O------------------------------------")
    try:
        res_h2o = (H2O(df, task, timelife))
        print(colored('Risultato memorizzato!', 'green'))
    except Exception as e:
        print(colored('Qualcosa è andato storto :(     -> ' + str(e), 'red'))
    print("------------------------------------H2O------------------------------------\n\n")
    return res_h2o


def fun_autogluon(df, task, timelife):
    res_autogluon = (0.0, 0.0)
    print("----------------------------------AUTOGLUON--------------------------------")
    try:
        res_autogluon = (autogluon(df, task, timelife))
        print(colored('Risultato memorizzato!', 'green'))
    except Exception as e:
        print(colored('Qualcosa è andato storto :(     -> ' + str(e), 'red'))
    print("----------------------------------AUTOGLUON--------------------------------\n\n")
    return res_autogluon