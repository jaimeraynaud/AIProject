# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 18:04:37 2021

@author: Jaime
"""
import calendar
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from scipy import stats


data = pd.read_csv("../data/train.csv")

print('Cantidad de Filas y columnas:',data.shape)
print('Nombre de las features:\n',data.columns)

'''Para comenzar a analizar los datos hacemos print de las variables y las primeras 5 filas de datos'''
print(data.head(5))

#Tipo de los distintos datos que aparecen:
print(data.info()) #Podemos observar que no hay valores nulos

#Pandas filtra las features numericas y calcula datos estadísticos que pueden ser útiles: 
#cantidad, media, desvío estandar, valores máximo y mínimo
print(data.describe())


#Antes de mostrar gráficas vamos a adaptar las features para un mejor entendimiento, usando Feature Engineering

data['hour'] = data.loc[:,'datetime'].dt.hour
data['day'] = data.loc[:,'datetime'].dt.weekday
data['month'] = data.loc[:,'datetime'].dt.month
data['year'] = data.loc[:,'datetime'].dt.year

#Eliminamos la columna datetime ya que ya no nos es necesaria
data = data.drop(["datetime"],axis=1)

data["season"] = data.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
data["weather"] = data.weather.map({1: "Good",\
                                              2 : "Cloudy", \
                                              3 : "Bad", \
                                              4 :"Very bad" })

#Pasamos a tipo category las categorical features
data[['season','holiday','workingday','weather', 'year','month','day','hour']] = data[['season','holiday','workingday','weather', 'year','month','day','hour']].astype('category')



#Ahora veamos las boxplot de count:
sns.boxplot(data=data,y="count",orient="v")

#Veamos si hay correlacion entre las variables:
corr = data.set_index('count').corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()

#Al ver que atemp y temp están altamente correlacionadas eliminaremos una de ellas.
data = data.drop('atemp',axis=1) 

#Outliers
Q1 = data["count"].quantile(0.25)
Q3 = data["count"].quantile(0.75)
IQR = Q3 - Q1

print("Shape con outliers: ", data.shape) 
  
# Upper bound
upper = np.where(data['count'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(data['count'] <= (Q1-1.5*IQR))
  
''' Eliminamos los outliers '''
data.drop(upper[0], inplace = True)
data.drop(lower[0], inplace = True)
  
print("Shape sin outliers: ", data.shape)

'''Veamos varios diagramas de barras para cada una de las features de nuestro conjunto'''
sns.factorplot(x='season',data=data,kind='count',size=5,aspect=2)
sns.factorplot(x='holiday',data=data,kind='count',size=5,aspect=2)
sns.factorplot(x='workingday',data=data,kind='count',size=5,aspect=2)
sns.factorplot(x='weather',data=data,kind='count',size=5,aspect=2) 

fig,axes=plt.subplots(2,2)
axes[0,0].hist(x="temp",data=data,edgecolor="black",linewidth=2)
axes[0,0].set_title("Temp")
axes[0,1].hist(x="atemp",data=data,edgecolor="black",linewidth=2)
axes[0,1].set_title("Atemp")
axes[1,0].hist(x="windspeed",data=data,edgecolor="black",linewidth=2)
axes[1,0].set_title("Windspeed")
axes[1,1].hist(x="humidity",data=data,edgecolor="black",linewidth=2)
axes[1,1].set_title("Humidity")
fig.set_size_inches(10,10)







