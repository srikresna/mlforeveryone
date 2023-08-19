from mlforeveryone.recognize import utils_recognize_type
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



def cross_distributions(dtf, x1, x2, y, max_cat=20, figsize=(10,5)):
    '''
    plots multivariate analysis.
    '''

    ## Y cat
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        
        ### cat vs cat --> contingency table
        if (utils_recognize_type(dtf, x1, max_cat) == "cat") & (utils_recognize_type(dtf, x2, max_cat) == "cat"):
            cont_table = pd.crosstab(index=dtf[x1], columns=dtf[x2], values=dtf[y], aggfunc="sum")
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cont_table, annot=True, fmt='.0f', cmap="YlGnBu", ax=ax, linewidths=.5).set_title(x1+'  vs  '+x2+'  (filter: '+y+')')
    
        ### num vs num --> scatter with hue
        elif (utils_recognize_type(dtf, x1, max_cat) == "num") & (utils_recognize_type(dtf, x2, max_cat) == "num"):
            sns.lmplot(x=x1, y=x2, data=dtf, hue=y, height=figsize[1])
        
        ### num vs cat --> boxplot with hue
        else:
            if (utils_recognize_type(dtf, x1, max_cat) == "cat"):
                cat,num = x1,x2
            else:
                cat,num = x2,x1
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(x=cat, y=num, hue=y, data=dtf, ax=ax).set_title(x1+'  vs  '+x2+'  (filter: '+y+')')
            ax.grid(True)
    
    ## Y num
    else:
        ### all num --> 3D scatter plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')  # Use add_subplot for 3D axes
        plot3d = ax.scatter(xs=dtf[x1], ys=dtf[x2], zs=dtf[y], c=dtf[y], cmap='inferno', linewidth=0.5)
        fig.colorbar(plot3d, shrink=0.5, aspect=5, label=y)
        ax.set_xlabel(x1)
        ax.set_ylabel(x2)
        ax.set_zlabel(y)
        plt.show()