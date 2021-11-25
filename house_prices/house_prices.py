# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:35:20 2021

@author: Alex
"""
#libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer as mkt
from sklearn.compose import make_column_selector as mks
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# import data -----------------------------------------------------------------------------------
X = pd.read_csv('train.csv')
y = X['SalePrice']
X.drop(['Id', 'SalePrice'], axis=1, inplace = True)
test = pd.read_csv('test.csv')
ids = test['Id']
test.drop(['Id'], axis = 1, inplace = True)

# inspect training data ------------------------------------------------------------------------
X.info()
X.isnull().sum()

#drop anything with more than 250 NaNs
# X.dropna(axis=1, thresh=len(X)-250, inplace=True)
# test.drop([c for c in test.columns if c not in X.columns], axis = 1, inplace=True)

# feature engineering -------------------------------------------------------------------------
knn_imp = KNNImputer(n_neighbors=5)
simp_imp = SimpleImputer(strategy = 'most_frequent')
ohe = OneHotEncoder(handle_unknown='ignore')

num_transf = make_pipeline(knn_imp) #impute missing values using KNN
cat_transf = make_pipeline(simp_imp, ohe)#impute missing values using median and one hot encode

#pipeline -------------------------------------------------------------------------------------

preproc = mkt((num_transf, mks(dtype_include='number')),
              (cat_transf, mks(dtype_exclude='number')))


#model--------------------------------------------------------------------------------------
rf = make_pipeline(preproc, RandomForestRegressor())
xgb = make_pipeline(preproc, XGBRegressor())
lr = make_pipeline(preproc, LinearRegression())

#hyperparameter tuning and fit on training data
rf_grid = {
    'randomforestregressor__n_estimators': [200, 500],
    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
    'randomforestregressor__max_depth' : [4,5,6,7,8]}
    
xgb_grid = {
    "xgbregressor__eta" : [0.05, 0.1, 0.2, 0.5], 
    'xgbregressor__n_estimators': [3, 5, 8],
    "xgbregressor__subsample": [0.7, 0.8, 0.9, 1.0],
    "xgbregressor__max_depth" : [2,3,4,6]}


rf = GridSearchCV(rf, rf_grid, cv = 3, n_jobs= 6, verbose=1, refit = True, scoring = "neg_mean_squared_error")
xgb = GridSearchCV(xgb, xgb_grid, cv = 3, n_jobs= 6, verbose=1, refit = True, scoring = "neg_mean_squared_error")

rf.fit(X, y)
xgb.fit(X,y)
lr.fit(X,y)

# predict on test set and create submission file------------------------------------------------
preds = pd.DataFrame({'rf': rf.predict(test), 'xgb': xgb.predict(test), 'lr': lr.predict(test)})
preds['maj'] = (preds['rf']+preds['xgb']+preds['lr'])/3
subm = pd.DataFrame({'Id': ids, 'SalePrice': preds['maj']})
subm.to_csv('sub.csv', index = False)


























