#!/usr/bin/env python3

from functions.openml_benchmark import openml_benchmark
from functions.kaggle_benchmark import kaggle_benchmark
from functions.test import test
import sys
import argparse
import os


def main():
    print("---------------------------------------START---------------------------------------")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndfopenml", help="number of opneml's dataframes to test")
    parser.add_argument("--dfkaggle", nargs="+", help="list of kaggle's dataframes to test: " + str(os.listdir('./datasets/kaggle')))
    parser.add_argument("--id", help="id of dataset to test")
    parser.add_argument("--algo", help="algorithm to use: autosklearn, tpot, h2o, autokeras, autogluon and all")
    args = parser.parse_args()

    if args.id and args.algo and len(sys.argv)-1 == 4:
        #print('test')
        test(int(args.id), args.algo)
    elif args.ndfopenml and len(sys.argv)-1 == 2:
        #print('openml')
        if int(args.ndfopenml) > 0:
            openml_benchmark(int(args.ndfopenml))
        else:
            print('Inserisci un numero positivo')
    elif args.dfkaggle and len(sys.argv) > 2:
        #print('kaggle ' + str(args.dfkaggle))
        kaggle_benchmark(args.dfkaggle)
    else:
        print('Comando non valido!')



if __name__ == '__main__':
    main()
        


    '''try:
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
        test()'''