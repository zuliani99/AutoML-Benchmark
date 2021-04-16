#!/usr/bin/env python3

# AUTOKERAS E AUTOGLUON -> scrivono nel disco per salvare tutti i modelli
# Ora tutti gli algoritmi sono parallelizzati, utilizzano tutti i core disponibili


from functions.openml_benchmark import openml_benchmark
from functions.kaggle_benchmark import kaggle_benchmark
from functions.test import test
import sys


def main():
    print("---------------------------------------START---------------------------------------")

    try:
        param = (sys.argv[1])
        if param.isnumeric():
            if int(param) > 0:
                openml_benchmark(int(param))
            else:
                print('Inserisci un numero positivo oppure non inserire nulla per eseguire un test singolo')
        else:
            if param =='kaggle':
                kaggle_benchmark()
            else:
                print('Comando non valido!')
    except IndexError:
        test()


if __name__ == '__main__':
    main()
        