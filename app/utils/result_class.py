import pandas as pd
from datetime import datetime
import os
from algorithms.auto_keras import autokeras
from algorithms.auto_sklearn import auto_sklearn
from algorithms.auto_gluon import autogluon
from algorithms.tpot import TPOT
from algorithms.h2o import H2O

class Result:
    def __init__(self, t):
        self.t = t
        self.res_class_acc = pd.DataFrame({'dataframe': [], 'autosklearn-acc': [], 'tpot-acc': [], 'h2o-acc': [], 'autokeras-acc': [], 'autogluon-acc': []})
        self.res_class_f1 = pd.DataFrame({'dataframe': [], 'autosklearn-f1': [], 'tpot-f1': [], 'h2o-f1': [], 'autokeras-f1': [], 'autogluon-f1': []})

        self.res_reg_rmse = pd.DataFrame({'dataframe': [], 'autosklearn-rmse': [], 'tpot-rmse': [], 'h2o-rmse': [], 'autokeras-rmse': [],'autogluon-rmse': []})
        self.res_reg_r2 = pd.DataFrame({'dataframe': [], 'autosklearn-r2': [], 'tpot-r2': [], 'h2o-r2': [], 'autokeras-r2': [], 'autogluon-r2': []})

        self.pipelines_class = pd.DataFrame({'dataframe': [], 'autosklearn': [], 'tpot': [], 'h2o': [], 'autokeras': [], 'autogluon': []})
        self.pipelines_reg = pd.DataFrame({'dataframe': [], 'autosklearn': [], 'tpot': [], 'h2o': [], 'autokeras': [], 'autogluon': []})

        self.options = None


    def run_benchmark(self, df, task, df_name, leader, options):
        res_as = auto_sklearn(df, task, options['autosklearn'])
        res_t = TPOT(df, task, options['tpot'])
        res_h = H2O(df, task, options['h2o'])
        res_ak = autokeras(df, task, options['autokeras'])
        res_ag = autogluon(df, task, options['autogluon'])

        self.options = pd.DataFrame({
            'autosklearn': [options['autosklearn']['time'], res_as[3]],
            'tpot': [options['tpot']['time'], res_t[3]],
            'h2o': [options['h2o']['time'], res_h[3]],
            'autokeras': [options['autokeras']['time'], res_ak[3]],
            'autogluon': [options['autogluon']['time'], res_ag[3]]
        })

        if (task == 'classification'):
            if leader is None:
                new_row_acc = {'dataframe': df_name, 'autosklearn-acc': res_as[0], 'tpot-acc': res_t[0], 'h2o-acc': res_h[0], 'autokeras-acc': res_ak[0], 'autogluon-acc': res_ag[0]}
                new_row_f1 = {'dataframe': df_name, 'autosklearn-f1': res_as[1], 'tpot-f1': res_t[1], 'h2o-f1': res_h[1], 'autokeras-f1': res_ak[1], 'autogluon-f1': res_ag[1]}

            elif (leader['measure'] == 'acc'):
                new_row_acc = {'dataframe': df_name, 'autosklearn-acc': res_as[0], 'tpot-acc': res_t[0], 'h2o-acc': res_h[0], 'autokeras-acc': res_ak[0], 'autogluon-acc': res_ag[0], 'leader': leader['score']}
                new_row_f1 = {'dataframe': df_name, 'autosklearn-f1': res_as[1], 'tpot-f1': res_t[1], 'h2o-f1': res_h[1], 'autokeras-f1': res_ak[1], 'autogluon-f1': res_ag[1], 'leader': None}
                print(new_row_acc)
            else:
                new_row_acc = {'dataframe': df_name, 'autosklearn-acc': res_as[0], 'tpot-acc': res_t[0], 'h2o-acc': res_h[0], 'autokeras-acc': res_ak[0], 'autogluon-acc': res_ag[0], 'leader': None}
                new_row_f1 = {'dataframe': df_name, 'autosklearn-f1': res_as[1], 'tpot-f1': res_t[1], 'h2o-f1': res_h[1], 'autokeras-f1': res_ak[1], 'autogluon-f1': res_ag[1], 'leader': leader['score']}
                print(new_row_f1)
            new_row_pipelines_class = {'dataframe': df_name, 'autosklearn': res_as[2], 'tpot': res_t[2], 'h2o': res_h[2], 'autokeras': res_ak[2], 'autogluon': res_ag[2]} # le pipeline sono già componenti html o dcc

            self.res_class_acc = self.res_class_acc.append(new_row_acc, ignore_index=True)
            self.res_class_f1 = self.res_class_f1.append(new_row_f1, ignore_index=True)
            self.pipelines_class = self.pipelines_class.append(new_row_pipelines_class, ignore_index=True)
        else:
            if leader is None:
                new_row_rmse = {'dataframe': df_name, 'autosklearn-rmse': res_as[0], 'tpot-rmse': res_t[0], 'h2o-rmse': res_h[0], 'autokeras-rmse': res_ak[0], 'autogluon-rmse': res_ag[0]}
                new_row_r2 = {'dataframe': df_name, 'autosklearn-r2': res_as[1], 'tpot-r2': res_t[1], 'h2o-r2': res_h[1], 'autokeras-r2': res_ak[1], 'autogluon-r2': res_ag[1]}

            elif (leader['measure'] == 'rmse'):
                new_row_rmse = {'dataframe': df_name, 'autosklearn-rmse': res_as[0], 'tpot-rmse': res_t[0], 'h2o-rmse': res_h[0], 'autokeras-rmse': res_ak[0], 'autogluon-rmse': res_ag[0], 'leader': leader['score']}
                new_row_r2 = {'dataframe': df_name, 'autosklearn-r2': res_as[1], 'tpot-r2': res_t[1], 'h2o-r2': res_h[1], 'autokeras-r2': res_ak[1], 'autogluon-r2': res_ag[1], 'leader': None}
                print(new_row_rmse)
            else:
                new_row_rmse = {'dataframe': df_name, 'autosklearn-rmse': res_as[0], 'tpot-rmse': res_t[0], 'h2o-rmse': res_h[0], 'autokeras-rmse': res_ak[0], 'autogluon-rmse': res_ag[0], 'leader': None}
                new_row_r2 = {'dataframe': df_name, 'autosklearn-r2': res_as[1], 'tpot-r2': res_t[1], 'h2o-r2': res_h[1], 'autokeras-r2': res_ak[1], 'autogluon-r2': res_ag[1], 'leader': leader['score']}
                print(new_row_r2)
            new_row_pipelines_reg = {'dataframe': df_name, 'autosklearn': res_as[2], 'tpot': res_t[2], 'h2o': res_h[2], 'autokeras': res_ak[2], 'autogluon': res_ag[2]} # sono componenti html o dcc

            self.res_reg_rmse = self.res_reg_rmse.append(new_row_rmse, ignore_index=True)
            self.res_reg_r2 = self.res_reg_r2.append(new_row_r2, ignore_index=True)
            self.pipelines_reg = self.pipelines_reg.append(new_row_pipelines_reg, ignore_index=True)


    def print_res(self):
        date = datetime.now()
        if(not self.res_class_acc.empty and not self.res_class_f1.empty):
            pathcla = './results/' + self.t + '/' + str(date).replace(' ', '-') + '/classification'
            os.makedirs(pathcla)
            print('---------------------------------RISULTATI DI CLASSIFICAZIONE ' + self.t + '---------------------------------')
            print(self.res_class_acc)
            print(self.res_class_f1)
            #print(self.pipelines_class)

            self.res_class_acc.to_csv(pathcla + '/acc.csv', index = False)
            self.res_class_f1.to_csv(pathcla + '/f1_score.csv', index = False)
            self.pipelines_class.to_csv(pathcla + '/pipelines.csv', sep='@', index = False)
            
        if(not self.res_reg_rmse.empty and not self.res_reg_r2.empty):
            pathreg = './results/' + self.t + '/' + str(date).replace(' ', '-') + '/regression'
            os.makedirs(pathreg)
            print('\n\n---------------------------------RISULTATI DI REGRESSIONE ' + self.t +'---------------------------------')
            print(self.res_reg_rmse)
            print(self.res_reg_r2)
            #print(self.pipelines_reg)

            self.res_reg_rmse.to_csv(pathreg + '/rmse.csv', index = False)
            self.res_reg_r2.to_csv(pathreg + '/r2_score.csv', index = False)
            self.pipelines_reg.to_csv(pathreg + '/pipelines.csv', sep='@', index = False)

        self.options.to_csv('./results/' + self.t + '/' + str(date).replace(' ', '-')+ '/options.csv', index = False)


        # Ritorno i dataframe oppure None se sono vuoti, ritorna una una lista di 4 dataframe
        return (str(date).replace(' ', '-'))