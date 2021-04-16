from sklearn.datasets import fetch_openml
from algorithms.auto_sklearn import auto_sklearn
from algorithms.tpot import TPOT
from algorithms.h2o import H2O
from algorithms.auto_keras import autokeras
from algorithms.auto_gluon import autogluon

def test():
    task = 'regression'

    id = 344
    X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True, cache=True)
    y = y.to_frame()
    X[y.columns[0]] = y
    df = X

    print(auto_sklearn(df, task))
    '''res = [fun_autosklearn(df, task), fun_tpot(df, task), fun_autokeras(df, task), fun_h2o(df), fun_autogluon(d, task)]

    new_row = {'dataset': str(id) + '.csv', 'autosklearn': res[0],'tpot': res[1], 'autokeras': res[2], 'h2o': res[3], 'autogluon': res[4], 'best': res_class.columns[np.argmax(res)+1] }


    res_class = res_class.append(new_row, ignore_index=True)
    print(res_class)

    res_reg = res_reg.append(new_row, ignore_index=True)
    print(res_reg)'''
        
    # autosklearn -> root_mean_squared_error: 0.7424714465226021
    # tpot -> reg:squarederror: -120.33836282299288
    # autokeras -> mean_squared_error: 0.006891193334013224 -> fixato
    # h2o -> mean_squared_error: 0.11184497546233797 -> fixato
    # autogluon -> root_mean_squared_error: 0.029061951526943217
    
