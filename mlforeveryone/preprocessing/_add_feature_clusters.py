import pandas as pd
import numpy as np

def add_feature_clusters(dtf, x, dic_clusters_mapping, dropx=False):
    '''
    Reduces the classes a categorical column.
    :parameter
        :param dtf: dataframe - feature matrix dtf
        :param x: str - column name
        :param dic_clusters_mapping: dict - ex: {"min":[30,45,180], "max":[60,120], "mean":[]}  where the residual class must have an empty list
        :param dropx: logic - whether the x column should be dropped
    '''    

    dic_flat = {v:k for k,lst in dic_clusters_mapping.items() for v in lst}
    for k,v in dic_clusters_mapping.items():
        if len(v)==0:
            residual_class = k 
    dtf[x+"_cluster"] = dtf[x].apply(lambda x: dic_flat[x] if x in dic_flat.keys() else residual_class)
    if dropx == True:
        dtf = dtf.drop(x, axis=1)
    return dtf