import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble


def features_importance(X, y, X_names, model=None, task="classification", figsize=(10,10)):
    '''
    Computes features importance.
    :parameter
        :param X: array
        :param X_names: list
        :param model: model istance (after fitting)
        :param figsize: tuple - plot setting
    :return
        dtf with features importance
    '''    
    
    ## model
    if model is None:
        if task == "classification":
            model = ensemble.GradientBoostingClassifier()  
        elif task == "regression":
            model = ensemble.GradientBoostingRegressor()
    model.fit(X,y)
    print("--- model used ---")
    print(model)
    
    ## importance dtf
    importances = model.feature_importances_
    dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":X_names}).sort_values("IMPORTANCE", ascending=False)
    dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
    dtf_importances = dtf_importances.set_index("VARIABLE")
    
    ## plot
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
    fig.suptitle("Features Importance", fontsize=20)
    ax[0].title.set_text('variables')
    dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.show()
    return dtf_importances.reset_index()
