from mlforeveryone.plot import bivariate_plot as bp
import pandas as pd
import numpy as np

def nan_analysis(dtf, na_x, y, max_cat=20, figsize=(10,5)):
    '''
    Plots a bivariate analysis using Nan and not-Nan as categories.
    '''

    dtf_NA = dtf[[na_x, y]]
    dtf_NA[na_x] = dtf[na_x].apply(lambda x: "Value" if not pd.isna(x) else "NA")
    bp(dtf_NA, x=na_x, y=y, max_cat=max_cat, figsize=figsize)
