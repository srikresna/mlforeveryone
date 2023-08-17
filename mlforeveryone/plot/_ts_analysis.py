from mlforeveryone.recognize import utils_recognize_type
import pandas as pd
import numpy as np

def ts_analysis(dtf, x, y, max_cat=20, figsize=(10,5)):
    '''
    Plots a bivariate analysis with time variable.
    '''

    if utils_recognize_type(dtf, y, max_cat) == "cat":
        dtf_tmp = dtf.groupby(x)[y].sum()       
    else:
        dtf_tmp = dtf.groupby(x)[y].median()
    dtf_tmp.plot(title=y+" by "+x, figsize=figsize, grid=True)