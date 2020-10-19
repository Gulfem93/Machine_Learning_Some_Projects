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
from sklearn.metrics import r2_score #(R-square (R-Kare))

#%%
#Verilerin Okunması

veriler = pd.read_csv("insurance.csv")
X = veriler.iloc[:,0:6]
Y = veriler.iloc[:,6:]
x = X.values
y = Y.values

sex = veriler.iloc[:,1:2]
smoker = veriler.iloc[:,4:5]
region = veriler.iloc[:,5:6]
#%%
# Verilerin Ön işlenmesi
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

sex = le.fit_transform(sex)
smoker = le.fit_transform(smoker)
region = ohe.fit_transform(region).toarray()

#DataFrame Birleştirme
sex = pd.DataFrame(data = sex, index = range(1338), columns = ["sex"])
smoker = pd.DataFrame(data = smoker, index = range(1338), columns = ["smoker"])
region = pd.DataFrame(data = region, index = range(1338), columns = ["nourtheast", "nourthwest", "sourtheast", "southwest"])

veri = pd.concat([sex, smoker], axis = 1)
veri = pd.concat([veri, region], axis = 1)

X = pd.concat([veriler.iloc[:,0:1], veri], axis =1)
X = pd.concat([X, veriler.iloc[:, 2:4]], axis = 1)

bw = np.append(arr = np.ones((1338, 1)).astype(int), values = X, axis = 1)
x_l = np.array(X, dtype = float)

model = sm.OLS(Y, x_l).fit()
print(model.summary())

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)


X = X.values
Y = Y.values

#%%
#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

tahmin_lin = lin_reg.predict(X)

#%%
#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

# 2. dereceden polinomal regresyon
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)

tahmin_poly = lin_reg2.predict(x_poly)

# 4. dereceden polinomal regresyon
poly_reg2 = PolynomialFeatures(degree = 4)
x_poly2 = poly_reg2.fit_transform(X)

lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly2, Y)
tahmin_poly2 = lin_reg4.predict(x_poly2)

#%% 
#Support Vector Regression (SVR)
from sklearn.svm import SVR

sc = StandardScaler()
sc2 = StandardScaler()
svr_reg = SVR(kernel = 'rbf')

x_sc = sc.fit_transform(X)
y_sc = np.ravel(sc2.fit_transform(Y))

svr_reg.fit(x_sc, y_sc)

tahmin_SVR = svr_reg.predict(x_sc)

#%%
#Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state = 0)
tree_reg.fit(X, Y)

tahmin_tree = tree_reg.predict(X)

#%%
#Random Forrest Regression
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(X, Y.ravel())
tahmin_rf = rf_reg.predict(X)

# R-square (R - kare)
print("\n----------------------\n")
print("Linear Regression")
print(r2_score(Y, lin_reg.predict(X)))

print("\n2. dereceden polinomal regresyon")
print(r2_score(Y, lin_reg2.predict(x_poly)))

print("\n4. dereceden polinomal regresyon")
print(r2_score(Y, lin_reg4.predict(x_poly2)))

print("\nSupport Vector Regression (SVR)")
print(r2_score(y_sc, svr_reg.predict(x_sc)))

print("\nDecision Tree")
print(r2_score(Y, tree_reg.predict(X)))

print("\nRandom Forrest Regression")
print(r2_score(Y, rf_reg.predict(X)))
