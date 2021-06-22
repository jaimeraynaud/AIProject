# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:04:04 2021

@author: Jaime
"""
''' Este es el script Prediccion para el proyecto de IA, 
 realizado por sus miembros: Juan Emilio Ordoñez y Jaime Raynaud Sánchez'''
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score


def read_files():
    train = pd.read_csv("../data/train.csv", parse_dates = ["datetime"])
    test = pd.read_csv("../data/test.csv", parse_dates = ["datetime"])
    
    train['hour'] = train.loc[:,'datetime'].dt.hour
    train['day'] = train.loc[:,'datetime'].dt.weekday
    train['month'] = train.loc[:,'datetime'].dt.month
    train['year'] = train.loc[:,'datetime'].dt.year
    
    test['hour'] = test.loc[:,'datetime'].dt.hour
    test['day'] = test.loc[:,'datetime'].dt.weekday
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

    #Convertimos los datos a categorical para poder trabajar con ellos.
    train[['season','holiday','workingday','weather', 'year','month','day','hour']] = train[['season','holiday','workingday','weather', 'year','month','day','hour']].astype('category')
    test[['season','holiday','workingday','weather', 'year','month','day','hour']] = test[['season','holiday','workingday','weather', 'year','month','day','hour']].astype('category')    
    return train, test

train, test = read_files()
train_sin_outliers, test = feature_engineering(train, test)

def models(train):
    
    train = train.drop('casual',axis=1)
    train = train.drop('registered',axis=1)
    
    x = train.drop('count',axis=1)
    y = train['count']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
    
    '''Random Forest Model'''
    rf_model = RandomForestRegressor(n_estimators= 100)
    rf_model.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)
    
    print("RMSE para Random Forest: ", np.sqrt(mean_squared_error(y_test,y_pred_rf)))
    print("R2 para Random Forest: ", r2_score(y_test, y_pred_rf))
    
    '''Light Gradient Boost Model'''
    lgb_model = lgb.LGBMRegressor(n_estimators=4000,alpha=0.01)
    lgb_model.fit(x_train, y_train)
    y_pred_lgb = lgb_model.predict(x_test)
    
    print("RMSE para LGBM: ",np.sqrt(mean_squared_error(y_test,y_pred_lgb)))
    print("R2 para LGBM: ", r2_score(y_test, y_pred_lgb))
    
    '''Gradient Boost Model'''
    gb_model = GradientBoostingRegressor(n_estimators=4000,alpha=0.01);
    gb_model.fit(x_train, y_train)
    y_pred_gb = gb_model.predict(x_test)

    print("RMSE para GBM: ",np.sqrt(mean_squared_error(y_test,y_pred_gb)))
    print("R2 para GBM: ", r2_score(y_test, y_pred_gb))

    
models(train_sin_outliers)

def prediccion(train, test):
    x = train.drop('count',axis=1)
    y = train['count']
    y_log = np.log(y.astype(float))
    
    gb_model = GradientBoostingRegressor(n_estimators=4000,alpha=0.01);
    gb_model.fit(x, y_log)
    
    
    y_pred_gb = gb_model.predict(np.log(test))
    
    return np.exp(y_pred_gb)

#preds = prediccion(train_sin_outliers, test)
#print(preds)
