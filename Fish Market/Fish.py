
#Bu veri seti, balık pazarı satışlarında yaygın olarak kullanılan 7 farklı balık türünün kaydıdır. 
#Bu veri setiyle, makine dostu veriler kullanılarak öngörücü bir model gerçekleştirilebilir 
# ve balıkların ağırlığı tahmin edilebilir.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

veriler = pd.read_csv("Fish.csv")
x = pd.concat([veriler.iloc[:,:1], veriler.iloc[:,2:]], axis = 1)
y = veriler.iloc[:,1:2]

#veri ön işleme
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
ohe = OneHotEncoder()
le = LabelEncoder()

Species = le.fit_transform(x.iloc[:,:1])
Species = pd.DataFrame(data = Species, index = range(159), columns = ["Species"])
x = pd.concat([Species, x.iloc[:,1:]], axis = 1)

#Train-test
x_train, x_test, y_train, y_test = train_test_split(x.iloc[:,4:5], y, test_size = 0.2, random_state = 0)

#linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

x_t = x_train.sort_index()
y_t = y_train.sort_index()

lr.fit(x_t, y_t)
linear_predict = lr.predict(x_test)


plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_test, linear_predict, color = 'red')
plt.show()

#SVR
from sklearn.svm import SVR
sc = StandardScaler()
sc2 = StandardScaler()
svr_reg = SVR(kernel = 'rbf')

x_sc = sc.fit_transform(x_t)
y_sc = np.ravel(sc2.fit_transform(y_t))

svr_reg.fit(x_sc, y_sc)

plt.scatter(x_sc, y_sc, color = 'blue')
plt.plot(x_sc, svr_reg.predict(x_sc), color = 'red')
plt.title('Support Vector Regression (SVR)')
plt.show()

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
Dtree_regression = DecisionTreeRegressor(criterion= 'mse')

Dtree_regression.fit(x_t, y_t)
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_test, Dtree_regression.predict(x_test), color = 'red')
plt.title('Decision Tree')
plt.show()

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
RForrest_Reg = RandomForestRegressor(n_estimators = 50, criterion="mse", random_state=0)

RForrest_Reg.fit(x_t, y_t)
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_test, RForrest_Reg.predict(x_test), color = 'red')
plt.title('Random Forest Regressor')
plt.show()

# R-square (R - kare)
from sklearn.metrics import r2_score

print("\n----------------------\n")
print("Linear Regression")
print(r2_score(y_test, linear_predict))

print("\nSupport Vector Regression (SVR)")
print(r2_score(y_sc, svr_reg.predict(x_sc)))

print("\nDecision Tree")
print(r2_score(y_test, Dtree_regression.predict(x_test)))

print("\nRandom Forrest Regression")
print(r2_score(y_test, RForrest_Reg.predict(x_test)))






























