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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
import seaborn as sns
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
    
    train = train.drop('casual',axis=1)
    train = train.drop('registered',axis=1)
    
    return train, test

train, test = read_files()
train_sin_outliers, test = feature_engineering(train, test)

def models(train):
    
    
    
    x = train.drop('count',axis=1)
    y = train['count']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
    
    '''Random Forest Model'''
    parameters_rf = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [None,1,2,3,4],
    #'criterion': ['mse', 'mae'],
    }
    
    rf_model = RandomForestRegressor(random_state = 0)
    grid_rf = GridSearchCV(estimator=rf_model, param_grid=parameters_rf, cv=2)
    grid_rf.fit(x_train, y_train)
    print("Mejores parámetros para Random Forest: ",grid_rf.best_params_)
    
    rf_model_best = RandomForestRegressor(n_estimators=300, max_depth=None)
    rf_model_best.fit(x_train, y_train)
    y_pred_rf = rf_model_best.predict(x_test)
    
    print("RMSE para Random Forest: ", np.sqrt(mean_squared_error(y_test,y_pred_rf)))
    print("R2 para Random Forest: ", r2_score(y_test, y_pred_rf))
    
    '''Light Gradient Boost Model'''
    parameters_lgb = {
    'n_estimators': [2000, 3000, 4000],
    'alpha': [0.001, 0.01, 0.1],
    }
    lgb_model = lgb.LGBMRegressor(random_state = 0)
    grid_lgb = GridSearchCV(estimator=lgb_model, param_grid=parameters_lgb, cv=2)
    grid_lgb.fit(x_train, y_train)
    print("Mejores parámetros para LGB: ",grid_lgb.best_params_)
    
    lgb_model_best = lgb.LGBMRegressor(n_estimators=2000, alpha=0.001)
    lgb_model_best.fit(x_train, y_train)
    y_pred_lgb = lgb_model_best.predict(x_test)
    
    print("RMSE para LGBM: ",np.sqrt(mean_squared_error(y_test,y_pred_lgb)))
    print("R2 para LGBM: ", r2_score(y_test, y_pred_lgb))
    
    '''Gradient Boost Model'''
    parameters_gb = {
    'n_estimators': [2000, 3000, 4000],
    'alpha': [0.001, 0.01, 0.1],
    }
    gb_model = GradientBoostingRegressor(random_state = 0)
    grid_gb = GridSearchCV(estimator=gb_model, param_grid=parameters_gb, cv=2)
    grid_gb.fit(x_train, y_train)
    print("Mejores parámetros para GB: ",grid_gb.best_params_)
    
    gb_model_best = GradientBoostingRegressor(n_estimators=4000, alpha=0.001);
    gb_model_best.fit(x_train, y_train)
    y_pred_gb = gb_model_best.predict(x_test)

    print("RMSE para GBM: ",np.sqrt(mean_squared_error(y_test,y_pred_gb)))
    print("R2 para GBM: ", r2_score(y_test, y_pred_gb))

    
#models(train_sin_outliers)

def prediccion(train, test):
    x = train.drop('count',axis=1)
    y = train['count']
    y_log = np.log1p(y)
    
    #Usamos el modelo y parámetros con mejor RMSE y R2, en nuestro caso LGB:
    lgb_model_best = lgb.LGBMRegressor(n_estimators=2000, alpha=0.001)
    lgb_model_best.fit(x, y_log)
    
    
    y_pred_lgb = lgb_model_best.predict(test)
    sns.distplot(np.exp(y_pred_lgb),bins=50)
    return np.exp(y_pred_lgb)

preds = prediccion(train_sin_outliers, test)

df_final = pd.read_csv("../data/test.csv", parse_dates = ["datetime"])
df_final['count'] = preds.tolist()
df_a_presentar = df_final[['datetime', 'count']]
print(df_a_presentar.describe())

df_a_presentar.to_csv(r'C:\Users\Jaime\Desktop\IA\Proyecto\code\data\PrediccionIA.txt', sep = ',', index = False, header = ['datetime','count'])