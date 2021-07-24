# Import necessari
import pandas as pd
from datetime import datetime
import os
from algorithms.auto_keras import autokeras
from algorithms.auto_sklearn import auto_sklearn
from algorithms.auto_gluon import autogluon
from algorithms.tpot import TPOT
from algorithms.h2o import H2O

# Creazione della classe Result addetta all'esecuzione di tutti gli algoritmi per un determinato DataFrame
class Result:
    def __init__(self, t):
        # Derfinizione dei campi della classe
        self.t = t
        self.res_class_acc = pd.DataFrame({'dataframe': [], 'autosklearn-acc': [], 'tpot-acc': [], 'h2o-acc': [], 'autokeras-acc': [], 'autogluon-acc': []})
        self.res_class_f1 = pd.DataFrame({'dataframe': [], 'autosklearn-f1': [], 'tpot-f1': [], 'h2o-f1': [], 'autokeras-f1': [], 'autogluon-f1': []})

        self.res_reg_rmse = pd.DataFrame({'dataframe': [], 'autosklearn-rmse': [], 'tpot-rmse': [], 'h2o-rmse': [], 'autokeras-rmse': [],'autogluon-rmse': []})
        self.res_reg_r2 = pd.DataFrame({'dataframe': [], 'autosklearn-r2': [], 'tpot-r2': [], 'h2o-r2': [], 'autokeras-r2': [], 'autogluon-r2': []})

        self.pipelines_class = self.pipelines_reg = pd.DataFrame({'dataframe': [], 'autosklearn': [], 'tpot': [], 'h2o': [], 'autokeras': [], 'autogluon': []})
    
        self.options_start = self.options_end = None

    # Funzione addetta all'esecuzione degli algoritmi per un determinato DataFrame e acnhe all'aggiornamento dei campi della classe
    def run_benchmark(self, df, task, df_name, leader, options):
        # Esecuzione degli algoritmi
        res_as = auto_sklearn(df, task, options['autosklearn'])
        res_t = TPOT(df, task, options['tpot'])
        res_h = H2O(df, task, options['h2o'])
        res_ak = autokeras(df, task, options['autokeras'])
        res_ag = autogluon(df, task, options['autogluon'])

        self.options_start = pd.DataFrame({
            'autosklearn-min': [options['autosklearn']['time']],
            'tpot-min': [options['tpot']['time']],
            'h2o-min': [options['h2o']['time']],
            'autokeras-epochs': [options['autokeras']['time']],
            'autogluon-min': [options['autogluon']['time']]
        })

        self.options_end = pd.DataFrame({
            'autosklearn-min': [res_as[3]],
            'tpot-min': [res_t[3]],
            'h2o-min': [res_h[3]],
            'autokeras-epochs': [res_ak[3]],
            'autogluon-min': [res_ag[3]]
        })

        # Aggiornamento dei campi di calssificazione o regressione a seconda del tipo
        if (task == 'classification'):
            new_row_acc, new_row_f1, new_row_pipelines_class = populate_row(df_name, leader, res_as, res_t, res_h, res_ak, res_ag, ('acc', 'f1'))

            self.res_class_acc = self.res_class_acc.append(new_row_acc, ignore_index=True)
            self.res_class_f1 = self.res_class_f1.append(new_row_f1, ignore_index=True)
            self.pipelines_class = self.pipelines_class.append(new_row_pipelines_class, ignore_index=True)

        else:
            new_row_rmse, new_row_r2, new_row_pipelines_reg = populate_row(df_name, leader, res_as, res_t, res_h, res_ak, res_ag, ('rmse', 'r2'))

            self.res_reg_rmse = self.res_reg_rmse.append(new_row_rmse, ignore_index=True)
            self.res_reg_r2 = self.res_reg_r2.append(new_row_r2, ignore_index=True)
            self.pipelines_reg = self.pipelines_reg.append(new_row_pipelines_reg, ignore_index=True)



    # Funzione rivolta alla conversione dei DataFrame in file csv per il mantenimento dei risultati
    def print_res(self):
        date = datetime.now()
        if(not self.res_class_acc.empty and not self.res_class_f1.empty):
            pathcla = './results/' + self.t + '/' + str(date).replace(' ', '-') + '/classification'
            os.makedirs(pathcla)
            print('---------------------------------CLASSIFICATION RESULTS ' + self.t + '---------------------------------')
            print(self.res_class_acc)
            print(self.res_class_f1)

            self.res_class_acc.insert(0, 'date', date)
            self.res_class_f1.insert(0, 'date', date)
            self.pipelines_class.insert(0, 'date', date)

            self.res_class_acc.to_csv(pathcla + '/acc.csv', index = False)
            self.res_class_f1.to_csv(pathcla + '/f1_score.csv', index = False)
            self.pipelines_class.to_csv(pathcla + '/pipelines.csv', sep='@', index = False)
            
        if(not self.res_reg_rmse.empty and not self.res_reg_r2.empty):
            pathreg = './results/' + self.t + '/' + str(date).replace(' ', '-') + '/regression'
            os.makedirs(pathreg)
            print('\n\n---------------------------------REGRESSION RESULTS ' + self.t +'---------------------------------')
            print(self.res_reg_rmse)
            print(self.res_reg_r2)

            self.res_reg_rmse.insert(0, 'date', date)
            self.res_reg_r2.insert(0, 'date', date)
            self.pipelines_reg.insert(0, 'date', date)

            self.res_reg_rmse.to_csv(pathreg + '/rmse.csv', index = False)
            self.res_reg_r2.to_csv(pathreg + '/r2_score.csv', index = False)
            self.pipelines_reg.to_csv(pathreg + '/pipelines.csv', sep='@', index = False)

        self.options_start.insert(0, 'date', date)
        self.options_end.insert(0, 'date', date)
        self.options_start.to_csv('./results/' + self.t + '/' + str(date).replace(' ', '-')+ '/options_start.csv', index = False)
        self.options_end.to_csv('./results/' + self.t + '/' + str(date).replace(' ', '-')+ '/options_end.csv', index = False)

        # Unico paramentro di ritorno è il timestamp che sarà utile poi per la visaulizzazione dei risultati
        return (str(date).replace(' ', '-'))


# Funnzione d'appoggio per il salvataggio la creazione delle nuove righe da inserire nei rispettivi DataFrame
def populate_row(df_name, leader, res_as, res_t, res_h, res_ak, res_ag, s):
    if leader is None:
        new_row_1 = {'dataframe': df_name, 'autosklearn-'+s[0]: res_as[0], 'tpot-'+s[0]: res_t[0], 'h2o-'+s[0]: res_h[0], 'autokeras-'+s[0]: res_ak[0], 'autogluon-'+s[0]: res_ag[0]}
        new_row_2 = {'dataframe': df_name, 'autosklearn-'+s[1]: res_as[1], 'tpot-'+s[1]: res_t[1], 'h2o-'+s[1]: res_h[1], 'autokeras-'+s[1]: res_ak[1], 'autogluon-'+s[1]: res_ag[1]}

    elif (leader['measure'] == s[0]):
        new_row_1 = {'dataframe': df_name, 'autosklearn-'+s[0]: res_as[0], 'tpot-'+s[0]: res_t[0], 'h2o-'+s[0]: res_h[0], 'autokeras-'+s[0]: res_ak[0], 'autogluon-'+s[0]: res_ag[0], 'leader': leader['score']}
        new_row_2 = {'dataframe': df_name, 'autosklearn-'+s[1]: res_as[1], 'tpot-'+s[1]: res_t[1], 'h2o-'+s[1]: res_h[1], 'autokeras-'+s[1]: res_ak[1], 'autogluon-'+s[1]: res_ag[1], 'leader': 'No value'}

    else:
        new_row_1 = {'dataframe': df_name, 'autosklearn-'+s[0]: res_as[0], 'tpot-'+s[0]: res_t[0], 'h2o-'+s[0]: res_h[0], 'autokeras-'+s[0]: res_ak[0], 'autogluon-'+s[0]: res_ag[0], 'leader': 'No value'}
        new_row_2 = {'dataframe': df_name, 'autosklearn-'+s[1]: res_as[1], 'tpot-'+s[1]: res_t[1], 'h2o-'+s[1]: res_h[1], 'autokeras-'+s[1]: res_ak[1], 'autogluon-'+s[1]: res_ag[1], 'leader': leader['score']}

    new_row_pipelines = {'dataframe': df_name, 'autosklearn': res_as[2], 'tpot': res_t[2], 'h2o': res_h[2], 'autokeras': res_ak[2], 'autogluon': res_ag[2]} # le pipeline sono già componenti html o dcc

    return new_row_1, new_row_2, new_row_pipelines