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

df = pd.read_csv("../data/train.csv")

print('Cantidad de Filas y columnas:',df.shape)
print('Nombre de las features:\n',df.columns)

'''Para comenzar a analizar los datos hacemos print de las variables y las primeras 5 filas de datos'''
print(df.head(5))

#Tipo de los distintos datos que aparecen:
print(df.info()) #Podemos observar que no hay valores nulos

#Pandas filtra las features numericas y calcula datos estadísticos que pueden ser útiles: 
#cantidad, media, desvío estandar, valores máximo y mínimo
print(df.describe())


#Antes de mostrar gráficas vamos a adaptar las features para un mejor entendimiento, usando Feature Engineering

df["date"] = df.datetime.apply(lambda x : x.split()[0])
df["hour"] = df.datetime.apply(lambda x : x.split()[1].split(":")[0])

df["weekday"] = df.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
df["month"] = df.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
df["season"] = df.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
df["weather"] = df.weather.map({1: "Good",\
                                              2 : "Cloudy", \
                                              3 : "Bad", \
                                              4 :"Very bad" })
    
categoryVariableList = ["hour","weekday","month","season","weather","holiday","workingday"]
for var in categoryVariableList:
    df[var] = df[var].astype("category")

#Eliminamos la columna datetime ya que ya no nos es necesaria
df = df.drop(["datetime"],axis=1)

#Ahora veamos las boxplot de count:

sns.boxplot(data=df,y="count",orient="v")

#Veamos si hay correlacion entre las variables:
corr = df.set_index('count').corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()
#Al ver que atemp y temp están altamente correlacionadas eliminaremos una de ellas.
df = df.drop('atemp',axis=1) 

#Outliers
Q1 = df["count"].quantile(0.25)
Q3 = df["count"].quantile(0.75)
IQR = Q3 - Q1

print("Shape con outliers: ", df.shape) 
  
# Upper bound
upper = np.where(df['count'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df['count'] <= (Q1-1.5*IQR))
  
''' Eliminamos los outliers '''
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)
  
print("Shape sin outliers: ", df.shape)

#Veamos ahora la grafica de distribución de densidad de la variable count
fig,axes = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(12, 10)
sns.distplot(df["count"],ax=axes[0][0])
stats.probplot(df["count"], dist='norm', fit=True, plot=axes[0][1])

#Al aplicarle la funcion logaritmo nos queda mucho más parecida a una distribución normal
sns.distplot(np.log(df["count"]),ax=axes[1][0])
stats.probplot(np.log1p(df["count"]), dist='norm', fit=True, plot=axes[1][1])

sns.factorplot(x='season',data=df,kind='count',size=5,aspect=2)
sns.factorplot(x='holiday',data=df,kind='count',size=5,aspect=2)
sns.factorplot(x='workingday',data=df,kind='count',size=5,aspect=2)
sns.factorplot(x='weather',data=df,kind='count',size=5,aspect=2) 

fig,axes=plt.subplots(2,2)
axes[0,0].hist(x="temp",data=df,edgecolor="black",linewidth=2)
axes[0,0].set_title("Temp")
axes[0,1].hist(x="atemp",data=df,edgecolor="black",linewidth=2)
axes[0,1].set_title("Atemp")
axes[1,0].hist(x="windspeed",data=df,edgecolor="black",linewidth=2)
axes[1,0].set_title("Windspeed")
axes[1,1].hist(x="humidity",data=df,edgecolor="black",linewidth=2)
axes[1,1].set_title("Humidity")
fig.set_size_inches(10,10)







