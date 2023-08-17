import pandas as pd
import numpy as np
from sklearn import utils
import imblearn

def rebalance(dtf, y, balance=None,  method="random", replace=True, size=1):
    '''
    Rebalances a dataset with up-sampling and down-sampling.
    :parameter
        :param dtf: dataframe - feature matrix dtf
        :param y: str - column to use as target 
        :param balance: str - "up", "down", if None just prints some stats
        :param method: str - "random" for sklearn or "knn" for imblearn
        :param size: num - 1 for same size of the other class, 0.5 for half of the other class
    :return
        rebalanced dtf
    '''
    
    ## check
    print("--- situation ---")
    check = dtf[y].value_counts().to_frame()
    check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
    print(check)
    print("tot:", check[y].sum())

    ## sklearn
    if balance is not None and method == "random":
        ### set the major and minor class
        major = check.index[0]
        minor = check.index[1]
        dtf_major = dtf[dtf[y]==major]
        dtf_minor = dtf[dtf[y]==minor]

        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   randomly replicate observations from the minority class (Overfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123, n_samples=int(size*len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   randomly remove observations of the majority class (Underfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123, n_samples=int(size*len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

    ## imblearn
    if balance is not None and method == "knn":
        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   create synthetic observations from the minority class (Distortion risk)")
            smote = imblearn.over_sampling.SMOTE(random_state=123)
            dtf_balanced, y_values = smote.fit_sample(dtf.drop(y,axis=1), y=dtf[y])
            dtf_balanced[y] = y_values
       
        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   select observations that don't affect performance (Underfitting risk)")
            nn = imblearn.under_sampling.CondensedNearestNeighbour(random_state=123)
            dtf_balanced, y_values = nn.fit_sample(dtf.drop(y,axis=1), y=dtf[y])
            dtf_balanced[y] = y_values
        
    ## check rebalance
    if balance is not None:
        print("--- new situation ---")
        check = dtf_balanced[y].value_counts().to_frame()
        check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
        print(check)
        print("tot:", check[y].sum())
        return dtf_balanced