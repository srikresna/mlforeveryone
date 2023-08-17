import pandas as pd

def utils_recognize_type(dtf, col, max_cat=20):
    '''
    Recognize whether a column is numerical or categorical.
    :parameter
        :param dtf: dataframe - input data
        :param col: str - name of the column to analyze
        :param max_cat: num - max number of unique values to recognize a column as categorical
    :return
        "cat" if the column is categorical, "dt" if datetime, "num" otherwise
    '''
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    elif dtf[col].dtype in ['datetime64[ns]','<M8[ns]']:
        return "dt"
    else:
        return "num"