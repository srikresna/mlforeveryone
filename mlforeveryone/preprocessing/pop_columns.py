import pandas as pd
import numpy as np

def pop_columns(dtf, lst_cols, where="front"):
    '''
    Moves columns into a dtf.
    :parameter
        :param dtf: dataframe - input data
        :param lst_cols: list - names of the columns that must be moved
        :param where: str - "front" or "end"
    :return
        dtf with moved columns
    '''

    current_cols = dtf.columns.tolist()
    for col in lst_cols:    
        current_cols.pop( current_cols.index(col) )
    if where == "front":
        dtf = dtf[lst_cols + current_cols]
    elif where == "end":
        dtf = dtf[current_cols + lst_cols]
    return dtf
