# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:04:04 2021

@author: Jaime
"""
''' Este es el script Prediccion para el proyecto de IA, 
 realizado por sus miembros: Juan Emilio Ordoñez y Jaime Raynaud Sánchez'''
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import r2_score

def read_files():
    train = pd.read_csv("../data/train.csv", parse_dates = ["datetime"])
    test = pd.read_csv("../data/test.csv", parse_dates = ["datetime"])
    
    train['hour'] = train.loc[:,'datetime'].dt.hour
    train['day'] = train.loc[:,'datetime'].dt.day
    train['month'] = train.loc[:,'datetime'].dt.month
    train['year'] = train.loc[:,'datetime'].dt.year
    
    test['hour'] = test.loc[:,'datetime'].dt.hour
    test['day'] = test.loc[:,'datetime'].dt.day
    test['month'] = test.loc[:,'datetime'].dt.month
    test['year'] = test.loc[:,'datetime'].dt.year
    
    train = train.drop('datetime',axis=1)
    test = test.drop('datetime',axis=1)
    
    return train, test

def feature_engineering(train, test):
    
    train = train.drop('atemp',axis=1)
    test = test.drop('atemp',axis=1)
    
    Q1 = train["count"].quantile(0.25)
    Q3 = train["count"].quantile(0.75)
    IQR = Q3 - Q1
    # Upper bound
    upper = np.where(train['count'] >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(train['count'] <= (Q1-1.5*IQR))
  
    ''' Eliminamos los outliers '''
    train.drop(upper[0], inplace = True)
    train.drop(lower[0], inplace = True)
    
# =============================================================================
#     train['windspeed'] = train['windspeed'].replace(0,np.NaN)
#     test['windspeed'] = test['windspeed'].replace(0,np.NaN)
#     train['windspeed'] = train['windspeed'].interpolate()
#     test['windspeed'] = test['windspeed'].interpolate()
# =============================================================================
    #Convertimos los datos a categorical para poder trabajar con ellos.
    train[['season','holiday','workingday','weather', 'year','month','day','hour']] = train[['season','holiday','workingday','weather', 'year','month','day','hour']].astype('category')
    test[['season','holiday','workingday','weather', 'year','month','day','hour']] = test[['season','holiday','workingday','weather', 'year','month','day','hour']].astype('category')    
    return train, test

train, test = read_files()
train.info()
test.info()
train_sin_outliers, test = feature_engineering(train, test)
train_sin_outliers.info()

def rmse(y_actual, y_predicted):
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    return rmse

def work(train, test):
    print(train.describe())
    x_train,x_test,y_train,y_test=train_test_split(train.drop('count',axis=1),train['count'],test_size=0.25,random_state=42)

    regressor = RandomForestRegressor(n_estimators= 1000, max_depth= 15, random_state=0, min_samples_split= 5, n_jobs=-1)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    
    r = rmse(y_test, y_pred)
    rSquaredRF = r2_score(y_test, y_pred)
    print("RMSE para Random Forest: %f" % (r))
    print("R2 para Random Forest: %f" % (rSquaredRF))
    
    hyperparameters = { 'colsample_bytree': 0.725,  'learning_rate': 0.013,
                    'num_leaves': 56, 'reg_alpha': 0.754, 'reg_lambda': 0.071, 
                    'subsample': 0.523, 'n_estimators': 1093}
    model = lgb.LGBMRegressor(**hyperparameters)
    model.fit(x_train, y_train)
    predslgbm = model.predict(x_test)
    
    r2 = rmse(y_test, predslgbm)
    rSquaredLGB = r2_score(y_test, predslgbm)
    print("RMSE para LGBM: %f" % (r2))
    print("R2 para LGBM: %f" % (rSquaredLGB))
    
    gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01);
    gbm.fit(x_train, y_train)
    predsGBM = gbm.predict(x_test)
    r3 = rmse(y_test, predsGBM)
    rSquaredGB = r2_score(y_test, predsGBM)
    print("RMSE para GBM: %f" % (r3))
    print("R2 para GBM: %f" % (rSquaredGB))
    
    
    
work(train_sin_outliers, test)