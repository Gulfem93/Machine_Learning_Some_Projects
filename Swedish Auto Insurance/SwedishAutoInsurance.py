#Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%%
#Veriler
veriler = pd.read_excel('SwedishAutoInsurance.xls')

X = veriler.iloc[:,0:1].values
Y = veriler.iloc[:,1:2].values


#%%
#Ön işleme
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

#Standart Scaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
Y_train = sc.fit_transform(y_train)

#%%
#Linear Regressioon 
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

linear_predict = linear_reg.predict(x_test)

#Figure
linear_figure = plt.figure()
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, linear_predict, c = 'blue')

plt.ylabel('Y')
plt.xlabel('X')
plt.title('Linear Regression')
plt.show()

#%%
#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)

lin_poly = LinearRegression()
lin_poly.fit(x_poly, Y)

#Figure
poly_figure = plt.figure()
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_poly.predict(x_poly), color = 'blue')

plt.ylabel('Y')
plt.xlabel('X')
plt.title('Polynomial Regression')
plt.show()

#%%
#SVR
from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')

sc1 = StandardScaler()
sc2 = StandardScaler()

x_ölcekli = sc1.fit_transform(X)
y_ölcekli = np.ravel(sc2.fit_transform(Y))


svr_reg.fit(x_ölcekli, y_ölcekli)
svr_predict = svr_reg.predict(x_ölcekli)

#Figure
svc_figure = plt.figure()
plt.scatter(x_ölcekli, y_ölcekli, color = 'red')
plt.plot(x_ölcekli, svr_predict, color = 'blue')

plt.ylabel('Y')
plt.xlabel('X')
plt.title('SVR')
plt.show()


#%%
#Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtree_reg = DecisionTreeRegressor(random_state=0)

dtree_reg.fit(X, Y)
dtree_predict = dtree_reg.predict(X)

#Figure
DTree = plt.figure()
plt.scatter(X, Y, color = 'red')
plt.plot(X, dtree_predict, color = 'blue')

plt.ylabel('Y')
plt.xlabel('X')
plt.title('Decision Tree')
plt.show()


#%%
#Random Forrest
from sklearn.ensemble import RandomForestRegressor
RForrest_reg = RandomForestRegressor(criterion = 'mse', random_state = 0)

RForrest_reg.fit(X, Y)
RForrest_predict = RForrest_reg.predict(X)

#Figure
RForrest = plt.figure()
plt.scatter(X, Y, color = 'red')
plt.plot(X, RForrest_predict, color = 'blue')

plt.ylabel('Y')
plt.xlabel('X')
plt.title('Random Forrest')
plt.show()







