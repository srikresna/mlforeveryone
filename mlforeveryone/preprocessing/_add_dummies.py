import pandas as pd
import numpy as np

def add_dummies(dtf, x, dropx=False):
    '''
    Transforms a categorical column into dummy columns
    :parameter
        :param dtf: dataframe - feature matrix dtf
        :param x: str - column name
        :param dropx: logic - whether the x column should be dropped
    :return
        dtf with dummy columns added
    '''    

    dtf_dummy = pd.get_dummies(dtf[x], prefix=x, drop_first=True, dummy_na=False)
    dtf = pd.concat([dtf, dtf_dummy], axis=1)
    print( dtf.filter(like=x, axis=1).head() )
    if dropx == True:
        dtf = dtf.drop(x, axis=1)
    return dtf