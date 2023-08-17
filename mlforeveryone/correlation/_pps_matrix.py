import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as ppscore
import numpy as np

def pps_matrix(dtf, annotation=True, lst_filters=[], figsize=(10,5)):
    '''
    Computes the pps matrix.
    '''
    
    dtf_pps = ppscore.matrix(dtf) if len(lst_filters) == 0 else ppscore.matrix(dtf).loc[lst_filters]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_pps, vmin=0., vmax=1., annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    plt.title("predictive power score")
    return dtf_pps