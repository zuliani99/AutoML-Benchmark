import pandas as pd
from usefull_functions import hist
import os
from datetime import datetime

path = './results/' + 'OpenML'+ '/' + str(datetime.now())
print(path)
os.makedirs(path)


#data = pd.read_csv('/home/riccardo/Desktop/cacca2.csv')
#data['cacca'] = 12
#hist(data, 'Openml - Regressione')
#data.to_csv('diocan.csv', index = False)

