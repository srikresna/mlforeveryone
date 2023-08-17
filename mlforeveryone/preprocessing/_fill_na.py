import pandas as pd
import numpy as np
from mlforeveryone.utils import utils_recognize_type

def fill_na(dtf, x, value=None):
    '''
    Replace Na with a specific value or mean for numerical and mode for categorical. 
    '''

    if value is None:
        value = dtf[x].mean() if utils_recognize_type(dtf, x) == "num" else dtf[x].mode().iloc[0]
        print("--- Replacing Nas with:", value, "---")
        dtf[x] = dtf[x].fillna(value)
        return dtf, value
    else:
        print("--- Replacing Nas with:", value, "---")
        dtf[x] = dtf[x].fillna(value)
        return dtf
