import pandas as pd
from utils.usefull_functions import hist


data = pd.read_csv('./results/openml/2021-04-23 18:21:21.381440/regression.csv')
hist(data, 'Openml - Reressione')
