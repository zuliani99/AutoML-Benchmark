import pandas as pd
from datetime import datetime
import os
from utils.algo_functions import fun_autosklearn, fun_tpot, fun_h2o, fun_autokeras, fun_autogluon


class Result:
    def __init__(self, t):
        self.t = t
        self.res_class_acc = pd.DataFrame({'dataset': [], 'autosklearn-acc': [], 'tpot-acc': [], 'autokeras-acc': [],'h2o-acc': [], 'autogluon-acc': []})
        self.res_class_f1 = pd.DataFrame({'dataset': [], 'autosklearn-f1': [], 'tpot-f1': [], 'autokeras-f1': [], 'h2o-f1': [], 'autogluon-f1': []})

        self.res_reg_rmse = pd.DataFrame({'dataset': [], 'autosklearn-rmse': [], 'tpot-rmse': [], 'autokeras-rmse': [], 'h2o-rmse': [],'autogluon-rmse': []})
        self.res_reg_r2 = pd.DataFrame({'dataset': [], 'autosklearn-r2': [], 'tpot-r2': [], 'autokeras-r2': [], 'h2o-r2': [], 'autogluon-r2': []})

        self.pipelines_class = pd.DataFrame({'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [],'autogluon': []})
        self.pipelines_reg = pd.DataFrame({'dataset': [], 'autosklearn': [], 'tpot': [], 'autokeras': [], 'h2o': [],'autogluon': []})

        self.options = None


    def run_benchmark(self, df, task, df_name, leader, options):
        res_as = fun_autosklearn(df, task, options['autosklearn'])
        res_t = fun_tpot(df, task, options['tpot'])
        res_ak = fun_autokeras(df, task, options['autokeras'])
        res_h = fun_h2o(df, task, options['h2o'])
        res_ag = fun_autogluon(df, task, options['autogluon'])

        self.options = pd.DataFrame({'autosklearn': options['autosklearn'], 'h2o': options['h2o'], 'tpot': options['tpot'], 'autokeras': options['autokeras'], 'autogluon': options['autogluon']}, ignore_index=True)

        if(task == 'classification'):
            if(leader is not None):
                new_row_acc = {'dataset': df_name, 'autosklearn-acc': res_as[0], 'tpot-acc': res_t[0], 'autokeras-acc': res_ak[0], 'h2o-acc': res_h[0], 'autogluon-acc': res_ag[0], leader['name']: leader['score']}
            else:
                new_row_acc = {'dataset': df_name, 'autosklearn-acc': res_as[0], 'tpot-acc': res_t[0], 'autokeras-acc': res_ak[0], 'h2o-acc': res_h[0], 'autogluon-acc': res_ag[0]}
            new_row_f1 = {'dataset': df_name, 'autosklearn-f1': res_as[1], 'tpot-f1': res_t[1], 'autokeras-f1': res_ak[1], 'h2o-f1': res_h[1], 'autogluon-f1': res_ag[1]}
            new_row_pipelines = {'dataset': df_name, 'autosklearn': res_as[2], 'tpot': res_t[2], 'autokeras': res_ak[2], 'h2o': res_h[2],'autogluon': res_ag[2]}
            self.res_class_acc = self.res_class_acc.append(new_row_acc, ignore_index=True)
            self.res_class_f1 = self.res_class_f1.append(new_row_f1, ignore_index=True)
            self.pipelines_class = self.pipelines_class.append(new_row_pipelines, ignore_index=True)
        else:
            if(leader is not None):
                new_row_rmse = {'dataset': df_name, 'autosklearn-rmse': res_as[0], 'tpot-rmse': res_t[0], 'autokeras-rmse': res_ak[0], 'h2o-rmse': res_h[0], 'autogluon-rmse': res_ag[0], leader['name']: leader['score']}
            else:
                new_row_rmse = {'dataset': df_name, 'autosklearn-rmse': res_as[0], 'tpot-rmse': res_t[0], 'autokeras-rmse': res_ak[0], 'h2o-rmse': res_h[0], 'autogluon-rmse': res_ag[0]}
            new_row_r2 = {'dataset': df_name, 'autosklearn-r2': res_as[1], 'tpot-r2': res_t[1], 'autokeras-r2': res_ak[1], 'h2o-r2': res_h[1], 'autogluon-r2': res_ag[1]}
            new_row_pipelines = {'dataset': df_name, 'autosklearn': res_as[2], 'tpot': res_t[2], 'autokeras': res_ak[2], 'h2o': res_h[2],'autogluon': res_ag[2]}
            self.res_reg_rmse = self.res_reg_rmse.append(new_row_rmse, ignore_index=True)
            self.res_reg_r2 = self.res_reg_r2.append(new_row_r2, ignore_index=True)
            self.pipelines_reg = self.pipelines_reg.append(new_row_pipelines, ignore_index=True)


    def print_res(self):
        date = datetime.now()
        if(not self.res_class_acc.empty and not self.res_class_f1.empty):
            pathcla = './results/' + self.t + '/' + str(date).replace(' ', '-') + '/classification'
            os.makedirs(pathcla)
            print('---------------------------------RISULTATI DI CLASSIFICAZIONE ' + self.t + '---------------------------------')
            print(self.res_class_acc)
            print(self.res_class_f1)

            self.res_class_acc.to_csv(pathcla + '/acc.csv', index = False)
            self.res_class_f1.to_csv(pathcla + '/f1_score.csv', index = False)
            self.pipelines_class.to_csv(pathcla + '/pipelines.csv', index = False)
            
        if(not self.res_reg_rmse.empty and not self.res_reg_r2.empty):
            pathreg = './results/' + self.t + '/' + str(date).replace(' ', '-') + '/regression'
            os.makedirs(pathreg)
            print('\n\n---------------------------------RISULTATI DI REGRESSIONE ' + self.t +'---------------------------------')
            print(self.res_reg_rmse)
            print(self.res_reg_r2)

            self.res_reg_rmse.to_csv(pathreg + '/rmse.csv', index = False)
            self.res_reg_r2.to_csv(pathreg + '/r2_score.csv', index = False)
            self.pipelines_reg.to_csv(pathreg + '/pipelines.csv', index = False)

        self.options.to_csv('./results/' + self.t + '/' + str(date).replace(' ', '-')+ '/options.csv', index = False)

        # Ritorno i dataframe oppure None se sono vuoti, ritorna una una lista di 4 dataframe
        return self.res_class_acc if not self.res_class_acc.empty else None, self.res_class_f1 if not self.res_class_f1.empty else None, self.res_reg_rmse if not self.res_reg_rmse.empty else None, self.res_reg_r2 if not self.res_reg_r2.empty else None, self.options