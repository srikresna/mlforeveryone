import pandas as pd
import numpy as np
from sklearn import feature_selection, linear_model
import matplotlib.pyplot as plt
import seaborn as sns

def features_selection(dtf, y, top=10, task="classification", figsize=(20,10)):
    '''
    Performs features selections: by correlation (keeping the lowest p-value) and by lasso.
    :prameter
        :param dtf: dataframe - feature matrix dtf
        :param y: str - name of the dependent variable
        :param top: num - number of top features
        :param task: str - "classification" or "regression"
    :return
        dic with lists of features to keep.
    '''     

    try:
        dtf_X = dtf.drop(y, axis=1)
        feature_names = dtf_X.columns
        
        ## p-value (one way anova F-test)
        model = feature_selection.f_classif if task=="classification" else feature_selection.f_regression
        selector = feature_selection.SelectKBest(score_func=model, k=top).fit(dtf_X.values, dtf[y].values)
        pvalue_selected_features = feature_names[selector.get_support()]
        
        ## regularization (classif-->lasso (l1), regr-->ridge (l2))
        model = linear_model.LogisticRegression(C=1, penalty="l1", solver='liblinear') if task=="classification" else linear_model.Ridge(alpha=1.0, fit_intercept=True)
        selector = feature_selection.SelectFromModel(estimator=model, max_features=top).fit(dtf_X.values, dtf[y].values)
        regularization_selected_features = feature_names[selector.get_support()]
        
        ## plot
        dtf_features = pd.DataFrame({"features":feature_names})
        dtf_features["p_value"] = dtf_features["features"].apply(lambda x: "p_value" if x in pvalue_selected_features else "")
        dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in pvalue_selected_features else 0)
        dtf_features["regularization"] = dtf_features["features"].apply(lambda x: "regularization" if x in regularization_selected_features else "")
        dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in regularization_selected_features else 0)
        dtf_features["method"] = dtf_features[["p_value","regularization"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
        dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
        dtf_features["method"] = dtf_features["method"].apply(lambda x: x.split()[0]+" + "+x.split()[1] if len(x.split())==2 else x)
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), ax=ax, dodge=False)
               
        join_selected_features = list(set(pvalue_selected_features).intersection(regularization_selected_features))
        return {"p_value":pvalue_selected_features, "regularization":regularization_selected_features, "join":join_selected_features}
    
    except Exception as e:
        print("--- got error ---")
        print(e)