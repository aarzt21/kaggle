#libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


#data
train = pd.read_csv('train.csv')
X = train.drop('SalePrice', axis = 1, inplace=False)
y = train['SalePrice']

test = pd.read_csv('test.csv')
ids = test['Id']
test.drop('Id', axis = 1, inplace = True)
X.drop('Id', axis = 1, inplace = True)

#explore
X.info()
#NA's per feature
X.isnull().sum()

#drop features with more than 300 NaNs
X.dropna(axis='columns', thresh = len(X)-300, inplace=True)
test = test.loc[:, X.columns] #and drop the same features from test data


num_feats = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_feats = [col for col in X.columns if X[col].dtype not in ['int64', 'float64']]

#Preprocessing pipeline --------------------------------------------------------------------------------------
cat_transf = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))])

num_transf = Pipeline(steps=[('impute', SimpleImputer(strategy='mean')),('imputer', KNNImputer(n_neighbors=5))])

preprocessor = ColumnTransformer(transformers=[
        ('num', num_transf, num_feats),
        ('cat', cat_transf, cat_feats)])

#model with pipeline-------------------------------------------------------------------------
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', RandomForestRegressor())])

xgb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', XGBRegressor())])

lr = Pipeline(steps=[('preprocessor',preprocessor), 
                       ('regressor', LinearRegression())])

#tuning grid --------------------------------------------------------------------------------
rf_grid = { 
    'regressor__n_estimators': [200, 500],
    'regressor__max_features': ['auto', 'sqrt', 'log2'],
    'regressor__max_depth' : [4,5,6,7,8]}

xgb_grid = {
    "regressor__eta" : [0.05, 0.1, 0.2, 0.5], 
    'regressor__n_estimators': [3, 5, 8],
    "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
    "regressor__max_depth" : [2,3,4,6]}

#fit--------------------------------------------------------------------------------------
rf = GridSearchCV(rf, rf_grid, n_jobs= 6, verbose=1, refit = True, scoring = "neg_mean_squared_error")
xgb = GridSearchCV(xgb, rf_grid, n_jobs= 6, verbose=1, refit = True, scoring = "neg_mean_squared_error")

rf.fit(X,y)
xgb.fit(X,y)
lr.fit(X,y)


#predict-------------------------------------------------------------------------------------
preds = pd.DataFrame({'rf': rf.predict(test), 'xgb': xgb.predict(test), 'lr': lr.predict(test)})
preds['maj'] = (preds['rf']+preds['xgb']+preds['lr'])/3
subm = pd.DataFrame({'Id': ids, 'SalePrice': preds['maj']})
subm.to_csv('subm.csv', index = False)












