import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def corr_matrix(dtf, method="pearson", negative=True, lst_filters=[], annotation=True, figsize=(10,5)):    
    '''
    Computes the correlation matrix.
    :parameter
        :param dtf: dataframe - input data
        :param method: str - "pearson" (numeric), "spearman" (categorical), "kendall"
        :param negative: bool - if False it takes the absolute values of correlation
        :param lst_filters: list - filter rows to show
        :param annotation: logic - plot setting
    '''
        
    ## factorize
    dtf_corr = dtf.copy()
    for col in dtf_corr.columns:
        if dtf_corr[col].dtype == "O":
            print("--- WARNING: Factorizing", dtf_corr[col].nunique(),"labels of", col, "---")
            dtf_corr[col] = dtf_corr[col].factorize(sort=True)[0]
    ## corr matrix
    dtf_corr = dtf_corr.corr(method=method) if len(lst_filters) == 0 else dtf_corr.corr(method=method).loc[lst_filters]
    dtf_corr = dtf_corr if negative is True else dtf_corr.abs()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_corr, annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    plt.title(method + " correlation")
    return dtf_corr