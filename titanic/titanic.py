import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from helpers import to_cat
from helpers import maj_vote

train = pd.read_csv('C:/Users/axarz/Desktop/Python_Practice/kaggle/Titanic/train.csv')
test = pd.read_csv('C:/Users/axarz/Desktop/Python_Practice/kaggle/Titanic/test.csv')
ids = test['PassengerId']

#let's drop some variables 
to_drop = ['PassengerId', 'Parch', 'Ticket', 'Cabin', 'Name']
train = train.drop(to_drop, axis=1)
test = test.drop(to_drop, axis = 1)

train['AgeBin'] = 6
for i in range(6):
    train.loc[(train.Age >= 10*i) & (train.Age < 10*(i + 1)), 'AgeBin'] = i

test['AgeBin'] = 6
for i in range(6):
    test.loc[(test.Age >= 10*i) & (test.Age < 10*(i + 1)), 'AgeBin'] = i

to_categ = ['Survived', 'Pclass', 'Embarked', 'Sex', 'SibSp', 'AgeBin']
to_cat(to_categ, train)
#to_cat(to_categ, test)
cat_feats = train.select_dtypes(include=['category']).drop('Survived', axis=1).columns
num_feats = train.select_dtypes(include = ['float64', 'int64']).columns


#Preprocessing steps-----------------------------------------------------------------------
cat_transf = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))]
)

num_transf = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='mean'))]
            )

preprocessor = ColumnTransformer(transformers=[
        ('num', num_transf, num_feats),
        ('cat', cat_transf, cat_feats)
        ])

# models (combined with pipeline)------------------------------------------------------------.
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

xgb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', XGBClassifier())])

lr = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

#tuning grids ---------------------------------------------------------------------------------

rf_grid = { 
    'classifier__n_estimators': [200, 500],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth' : [4,5,6,7,8],
    'classifier__criterion' :['gini', 'entropy']}

xgb_grid = {
    "classifier__eta" : [0.05, 0.1, 0.2, 0.5], 
    'classifier__n_estimators': [3, 5, 8],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__max_depth" : [2,3,4]}


#CV and refit with optimal hyperparas---------------------------------------------------
rf = GridSearchCV(rf, rf_grid, n_jobs= 6, verbose=1, refit = True, scoring = "accuracy")
xgb = GridSearchCV(xgb, xgb_grid, n_jobs= 6, verbose=1, refit = True, scoring = "accuracy")

rf.fit(train.drop(['Survived'], axis = 1), train['Survived'])
xgb.fit(train.drop(['Survived'], axis = 1), train['Survived'])
lr.fit(train.drop(['Survived'], axis = 1), train['Survived'])

#combine predictions into majority vote prediction
preds = maj_vote(rf.predict(test), xgb.predict(test), lr.predict(test))

submission = pd.DataFrame({'PassengerID': ids, 'Survived': preds})
submission.to_csv('sub.csv', index = False)
