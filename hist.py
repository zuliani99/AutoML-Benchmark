import pandas as pd
from utils.usefull_functions import hist


data = pd.read_csv('./results/kaggle/2021-04-22 10:42:04.270430/classification.csv')
hist(data, 'Kaggel - Classificazione')
