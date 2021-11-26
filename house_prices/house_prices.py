
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer


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

# types of features to apply different transformations to
num_feats = list(X.select_dtypes(include=('number')))
log_feats = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
              'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
              'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 
              'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
              'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
              '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
num_feats = list(set(num_feats)-set(log_feats))
cat_feats = list(X.select_dtypes(exclude=('number')))

# transf pipeline for num feats
num_transf = Pipeline(steps=[ ('imp1',SimpleImputer())] )
# transf pipeline for log feats
log_transf = Pipeline(steps= 
                      [('imp2', num_transf), ('log', FunctionTransformer(np.log1p))])
# transf pipline for cat feats
cat_transf = Pipeline(steps =[
                      ('cat_imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))]
                      )

#pipelines applied selectively to their respective features
preprocessor = ColumnTransformer(transformers=[
    ('num_tranfs', num_transf, num_feats),
    ('log_tranfs', log_transf, log_feats),
    ('cat_tranfs', cat_transf, cat_feats)
    ], remainder = 'passthrough')

#combine preproc with model into one pipeline
rf = Pipeline(steps=[
    ('prepro', preprocessor),
    ('rf', RandomForestRegressor())
    ])

xgb =Pipeline(steps=[
    ('prepro', preprocessor),
    ('xgb', XGBRegressor())
    ])

lr =Pipeline(steps=[
    ('prepro', preprocessor),
    ('lr', LinearRegression())
    ])


#grid search and fit with optimal hyperparas
rf_grid = {
    'rf__n_estimators': [200, 500],
    'rf__max_features': ['auto', 'sqrt', 'log2'],
    'rf__max_depth' : [4,5,6,7,8]}
    
xgb_grid = {
    "xgb__eta" : [0.05, 0.1, 0.2, 0.5], 
    'xgb__n_estimators': [3, 5, 8],
    "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
    "xgb__max_depth" : [2,3,4,6]}


rf = GridSearchCV(rf, rf_grid, cv = 5, n_jobs= 6, verbose=1, refit = True, scoring = "neg_mean_squared_error")
xgb = GridSearchCV(xgb, xgb_grid, cv = 5, n_jobs= 6, verbose=1, refit = True, scoring = "neg_mean_squared_error")

rf.fit(X, y)
xgb.fit(X,y)
lr.fit(X,y)


# predict on test set and create submission file------------------------------------------------
preds = pd.DataFrame({'rf': rf.predict(test), 'xgb': xgb.predict(test), 'lr': lr.predict(test)})
preds['maj'] = (preds['rf']+preds['xgb']+preds['lr'])/3
subm = pd.DataFrame({'Id': ids, 'SalePrice': preds['maj']})
subm.to_csv('subi_full.csv', index = False)































