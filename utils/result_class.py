import pandas as pd
from datetime import datetime
import os
from utils.usefull_functions import scatter, hist
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon

class Result:
    def __init__(self, t):
        self.t = t
        self.res_class_acc = pd.DataFrame({'dataset': [], 'autosklearn-acc': [], 'tpot-acc': [], 'autokeras-acc': [],'h2o-acc': [], 'autogluon-acc': []})
        self.res_class_f1 = pd.DataFrame({'dataset': [], 'autosklearn-f1': [], 'tpot-f1': [], 'autokeras-f1': [], 'h2o-f1': [], 'autogluon-f1': []})

        self.res_reg_reme = pd.DataFrame({'dataset': [], 'autosklearn-rmse': [], 'tpot-rmse': [], 'autokeras-rmse': [], 'h2o-rmse': [],'autogluon-rmse': []})
        self.res_reg_r2 = pd.DataFrame({'dataset': [], 'autosklearn-r2': [], 'tpot-r2': [], 'autokeras-r2': [], 'h2o-r2': [], 'autogluon-r2': []})


    def run_benchmark(self, df, task, df_name):
        res_as = fun_autosklearn(df, task)
        res_t = fun_tpot(df, task)
        res_h = fun_autokeras(df, task)
        res_ak = fun_h2o(df, task)
        res_ag = fun_autogluon(df, task)

        if(task == 'classification'):
            new_row_acc = {'dataset': df_name, 'autosklearn-acc': res_as[0], 'tpot-acc': res_t[0], 'autokeras-acc': res_ak[0], 'h2o-acc': res_h[0], 'autogluon-acc': res_ag[0]}
            new_row_f1 = {'dataset': df_name, 'autosklearn-f1': res_as[1], 'tpot-f1': res_t[1], 'autokeras-f1': res_ak[1], 'h2o-f1': res_h[1], 'autogluon-f1': res_ag[1]}
            self.res_class_acc = self.res_class_acc.append(new_row_acc, ignore_index=True)
            self.res_class_f1 = self.res_class_f1.append(new_row_f1, ignore_index=True)
        else:
            new_row_rmse = {'dataset': df_name, 'autosklearn-rmse': res_as[0], 'tpot-rmse': res_t[0], 'autokeras-rmse': res_ak[0], 'h2o-rmse': res_h[0], 'autogluon-rmse': res_ag[0]}
            new_row_r2 = {'dataset': df_name, 'autosklearn-r2': res_as[1], 'tpot-r2': res_t[1], 'autokeras-r2': res_ak[1], 'h2o-r2': res_h[1], 'autogluon-r2': res_ag[1]}
            self.res_reg_rmse = self.res_reg_rmse.append(new_row_rmse, ignore_index=True)
            self.res_reg_r2 = self.res_reg_r2.append(new_row_r2, ignore_index=True)


    def print_res(self):
        if(not self.res_class_acc.empty and not self.res_class_f1.empty):
            pathcla = './results/' + self.t + '/' + str(datetime.now()).replace(' ', '-') + 'classification'
            os.makedirs(pathcla)
            print('---------------------------------RISULTATI DI CLASSIFICAZIONE ' + self.t + '---------------------------------')
            print(self.res_class_acc)
            print(self.res_class_f1)

            self.res_class_acc.to_csv(pathcla + '/acc.csv', index = False)
            self.res_class_f1.to_csv(pathcla + '/f1.csv', index = False)
            #hist(self.res_class, self.t + ' - Classificazione')
            
        if(not self.res_reg_rmse.empty and not self.res_reg.empty_r2):
            pathreg = './results/' + self.t + '/' + str(datetime.now()).replace(' ', '-') + 'regression'
            os.makedirs(pathreg)
            print('\n\n---------------------------------RISULTATI DI REGRESSIONE ' + self.t +'---------------------------------')
            print(self.res_reg_rmse)
            print(self.res_reg_r2)

            self.res_reg_rmse.to_csv(pathreg + '/rmse.csv', index = False)
            self.res_reg_r2.to_csv(pathreg + '/r2.csv', index = False)
            #hist(self.res_reg, self.t + ' - Regressione')
        
        # Ritorno i dataframe oppure None se sono vuoti, ritorn una una lista di 4 dataframe
        return self.res_class_acc if not self.res_class_acc.empty else None, self.res_class_f1 if not self.res_class_f1.empty else None,
        self.res_reg_rmse if not self.res_reg_rmse.empty else None, self.res_reg_r2 if not self.res_reg_r2.empty else None