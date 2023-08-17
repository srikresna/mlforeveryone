from mlforeveryone.recognize import utils_recognize_type
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def dtf_overview(dtf, max_cat=20, figsize=(10,5)):
    '''
    Get a general overview of a dataframe.
    :parameter
        :param dtf: dataframe - input data
        :param max_cat: num - mininum number of recognize column type
    '''

    ## recognize column type
    dic_cols = {col:utils_recognize_type(dtf, col, max_cat=max_cat) for col in dtf.columns}
        
    ## print info
    len_dtf = len(dtf)
    print("Shape:", dtf.shape)
    print("-----------------")
    for col in dtf.columns:
        info = col+" --> Type:"+dic_cols[col]
        info = info+" | Nas: "+str(dtf[col].isna().sum())+"("+str(int(dtf[col].isna().mean()*100))+"%)"
        if dic_cols[col] == "cat":
            info = info+" | Categories: "+str(dtf[col].nunique())
        elif dic_cols[col] == "dt":
            info = info+" | Range: "+"({x})-({y})".format(x=str(dtf[col].min()), y=str(dtf[col].max()))
        else:
            info = info+" | Min-Max: "+"({x})-({y})".format(x=str(int(dtf[col].min())), y=str(int(dtf[col].max())))
        if dtf[col].nunique() == len_dtf:
            info = info+" | Possible PK"
        print(info)
                
    ## plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = dtf.isnull()
    for k,v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Dataset Overview')
    #plt.setp(plt.xticks()[1], rotation=0)
    plt.show()
    
    ## add legend
    print("\033[1;37;40m Categerocial \033[m", "\033[1;30;41m Numerical/DateTime \033[m", "\033[1;30;47m NaN \033[m")