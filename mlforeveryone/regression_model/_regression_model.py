import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, model_selection, metrics
import minisom
from statsmodels.graphics.api import abline_plot
from tensorflow.keras import models, layers, backend as K

def fit_ml_regr(model, X_train, y_train, X_test, scalerY=None):
    '''
    Fits a sklearn regression model.
    :parameter
        :param model: model object - model to fit (before fitting)
        :param X_train: array
        :param y_train: array
        :param X_test: array
        :param scalerY: scaler object (only for regression)
    :return
        model fitted and predictions
    '''      
    ## model
    model = ensemble.GradientBoostingRegressor() if model is None else model
    
    ## train/test
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    if scalerY is not None:
        predicted = scalerY.inverse_transform(predicted.reshape(-1,1)).reshape(-1)
    return model, predicted




def tune_regr_model(X_train, y_train, model_base=None, param_dic=None, scoring="r2", searchtype="RandomSearch", n_iter=1000, cv=10, figsize=(10,5)):
    '''
    Tunes the hyperparameters of a sklearn regression model.
    '''

    model_base = ensemble.GradientBoostingRegressor() if model_base is None else model_base
    param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750], 'max_depth':[2,3,4,5,6,7]} if param_dic is None else param_dic                        

    ## Search
    print("---", searchtype, "---")
    if searchtype == "RandomSearch":
        random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=n_iter, scoring=scoring).fit(X_train, y_train)
        print("Best Model parameters:", random_search.best_params_)
        print("Best Model "+scoring+":", round(random_search.best_score_, 2))
        model = random_search.best_estimator_
    
    elif searchtype == "GridSearch":
        grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring=scoring).fit(X_train, y_train)
        print("Best Model parameters:", grid_search.best_params_)
        print("Best Model mean "+scoring+":", round(grid_search.best_score_, 2))
        model = grid_search.best_estimator_
    
    ## K fold validation
    print("")
    print("--- Kfold Validation ---")
    Kfold_base = model_selection.cross_validate(estimator=model_base, X=X_train, y=y_train, cv=cv, scoring=scoring)
    Kfold_model = model_selection.cross_validate(estimator=model, X=X_train, y=y_train, cv=cv, scoring=scoring)
    print(scoring, "mean - base model:", round(Kfold_base["test_score"].mean(),2), " --> best model:", round(Kfold_model["test_score"].mean()))
    
    scores = []
    cv = model_selection.KFold(n_splits=cv, shuffle=True)
    fig = plt.figure(figsize=figsize)
    i = 1
    for train, test in cv.split(X_train, y_train):
        prediction = model.fit(X_train[train], y_train[train]).predict(X_train[test])
        true = y_train[test]
        score = metrics.r2_score(true, prediction)
        scores.append(score)
        plt.scatter(prediction, true, lw=2, alpha=0.3, label='Fold %d (R2 = %0.2f)' % (i,score))
        i = i+1
    plt.plot([min(y_train),max(y_train)], [min(y_train),max(y_train)], linestyle='--', lw=2, color='black')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('K-Fold Validation')
    plt.legend()
    plt.show()
    
    return model




def evaluate_regr_model(y_test, predicted, figsize=(25,5)):
    '''
    Evaluates a model performance.
    :parameter
        :param y_test: array
        :param predicted: array
    '''    
    
    ## Kpi
    print("R2 (explained variance):", round(metrics.r2_score(y_test, predicted), 2))
    print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", round(np.mean(np.abs((y_test-predicted)/predicted)), 2))
    print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(metrics.mean_absolute_error(y_test, predicted)))
    print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
    
    ## residuals
    residuals = y_test - predicted
    max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
    max_true, max_pred = y_test[max_idx], predicted[max_idx]
    print("Max Error:", "{:,.0f}".format(max_error))
    
    ## Plot predicted vs true
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    ax[0].scatter(predicted, y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    ax[0].legend()
    
    ## Plot predicted vs residuals
    ax[1].scatter(predicted, residuals, color="red")
    ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
    ax[1].grid(True)
    ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
    ax[1].hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
    ax[1].legend()
    
    ## Plot residuals distribution
    sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax[2], label="mean = "+"{:,.0f}".format(np.mean(residuals)))
    ax[2].grid(True)
    ax[2].set(yticks=[], yticklabels=[], title="Residuals distribution")
    plt.show()
