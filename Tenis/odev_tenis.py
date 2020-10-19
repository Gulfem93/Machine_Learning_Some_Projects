#%%
#Kütüphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#%%
#Verilerin Okunması

veriler = pd.read_csv("odev_tenis.csv")

outlook = veriler.iloc[:,0:1]
play = veriler.iloc[:,4:]
windy = veriler.iloc[:,3:4]
temperature_humidity = veriler.iloc[:,1:3]
temperature = veriler.iloc[:,1:2]
humidity = veriler.iloc[:,2:3]

#%%
#Verilerin Ön İşlemesi

##Kategorize Edildi
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

outlook = ohe.fit_transform(outlook).toarray()
windy = ohe.fit_transform(windy).toarray()
play = ohe.fit_transform(play).toarray()

##Dataframe Dönüştürüldü
outlook = pd.DataFrame(data = outlook, index = range(14), columns = ["overcast", "rainy", "sunny"])
windy = pd.DataFrame(data = windy[:,0:1], index = range(14), columns = ["windy"])
play = pd.DataFrame(data = play[:,0:1], index = range(14), columns = ["play"])

##DataFrameler birleştirildi
veri = pd.concat([windy, play], axis = 1)
veri = pd.concat([veri, outlook], axis = 1)
veri = pd.concat([veri, temperature_humidity], axis = 1)

##Eğitilmesi (Train Test)
x_train, x_test, y_train, y_test = train_test_split(veri.iloc[:,:-1], veri.iloc[:,-1:], test_size = 0.33, random_state = 0)


#%%
#Tamin yaparım

lr = LinearRegression()
lr.fit(x_train, y_train)

##tahmin
tahmin = lr.predict(x_test)


#%%
#Bacward Elimination

X = np.append(arr = np.ones((14,1)).astype(int), values = veri.iloc[:,:-1], axis = 1)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)

model = sm.OLS(temperature_humidity.iloc[:,1:], X_l).fit()
print(model.summary())


veri = veri.iloc[:,1:]
X = np.append(arr = np.ones((14,1)).astype(int), values = veri.iloc[:,:-1], axis = 1)
X_l = veri.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l, dtype = float)

model = sm.OLS(temperature_humidity.iloc[:,1:], X_l).fit()
print(model.summary())

x_test = x_test.iloc[:,1:]
x_train = x_train.iloc[:,1:]

lr.fit(x_train, y_train)

##tahmin
tahmin1 = lr.predict(x_test)
