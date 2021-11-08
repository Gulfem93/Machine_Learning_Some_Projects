import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

house = pd.read_csv('kc_house_data.csv')

price = house.iloc[:,2:3].values
living = house.iloc[:,5:6].values

price = pd.DataFrame(data = price, index = range(21613), columns = ['price'])
living = pd.DataFrame(data = living, index = range(21613), columns = ['sqfl_living'])

lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(living, price, test_size = 1/3, random_state = 0)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

lr.fit(x_train, y_train)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_test, lr.predict(x_test))



