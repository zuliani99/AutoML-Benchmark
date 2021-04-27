import pandas as pd
from datetime import datetime
import os
from utils.usefull_functions import scatter, hist
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon

class Result:
    def __init__(self, t):
        self.t = t
        self.res_class = pd.DataFrame({'dataset': [], 'autosklearn-acc': [], 'autosklearn-f1': [],
                                        'tpot-acc': [], 'tpot-f1': [], 'autokeras-acc': [], 'autokeras-f1': [],
                                        'h2o-acc': [], 'h2o-f1': [], 'autogluon-acc': [], 'autogluon-f1': []})
        self.res_reg = pd.DataFrame({'dataset': [], 'autosklearn-rmse': [], 'autosklearn-r2': [],
                                    'tpot-rmse': [], 'tpot-r2': [], 'autokeras-rmse': [], 'autokeras-r2': [],
                                    'h2o-rmse': [], 'h2o-r2': [], 'autogluon-rmse': [], 'autogluon-r2': []})


    def run_benchmark(self, df, task, df_name):
        res_as = fun_autosklearn(df, task)
        res_t = fun_tpot(df, task)
        res_h = fun_autokeras(df, task)
        res_ak = fun_h2o(df, task)
        res_ag = fun_autogluon(df, task)

        if(task == 'classification'):
            new_row = {'dataset': df_name, 'autosklearn-acc': res_as[0], 'autosklearn-f1': res_as[1],
            'tpot-acc': res_t[0], 'tpot-f1': res_t[1], 'autokeras-acc': res_ak[0], 'autokeras-f1': res_ak[1],
            'h2o-acc': res_h[0], 'h2o-f1': res_h[1], 'autogluon-acc': res_ag[0], 'autogluon-f1': res_ag[1]}
            self.res_class = self.res_class.append(new_row, ignore_index=True)
        else:
            new_row = {'dataset': df_name, 'autosklearn-rmse': res_as[0], 'autosklearn-r2': res_as[1],
            'tpot-rmse': res_t[0], 'tpot-r2': res_t[1], 'autokeras-rmse': res_ak[0], 'autokeras-r2': res_ak[1],
            'h2o-rmse': res_h[0], 'h2o-r2': res_h[1], 'autogluon-rmse': res_ag[0], 'autogluon-r2': res_ag[1]}
            self.res_reg = self.res_reg.append(new_row, ignore_index=True)


    def print_res(self):
        path = './results/' + self.t + '/' + str(datetime.now())
        os.makedirs(path)
        if(not self.res_class.empty):
            print('---------------------------------RISULTATI DI CLASSIFICAZIONE ' + self.t + '---------------------------------')
            print(self.res_class)

            self.res_class.to_csv(path + '/classification.csv', index = False)
            hist(self.res_class, self.t + ' - Classificazione')
        if(not self.res_reg.empty):
            print('\n\n---------------------------------RISULTATI DI REGRESSIONE ' + self.t +'---------------------------------')
            print(self.res_reg)

            self.res_reg.to_csv(path + '/regression.csv', index = False)
            hist(self.res_reg, self.t + ' - Regressione')