import pandas as pd
import os
from datetime import datetime

path = './results/OpenML/2021-04-28-19:52:53.106617/classification/f1_score.csv'
print(path)
data = pd.read_csv(path)
print(len(data.columns))


#data = pd.read_csv('/home/riccardo/Desktop/cacca2.csv')
#data['cacca'] = 12
#hist(data, 'Openml - Regressione')
#data.to_csv('diocan.csv', index = False)

