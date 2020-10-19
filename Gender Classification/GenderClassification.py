#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib.colors import ListedColormap, BoundaryNorm
#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('GenderClassification.csv')
#pd.read_csv("veriler.csv")
#test

le = LabelEncoder()
ohe = OneHotEncoder()


objFeatures = veriler.select_dtypes(include="object").columns

for feat in objFeatures:
    veriler[feat] = le.fit_transform(veriler[feat].astype(str))

x = veriler.iloc[:,0:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken

veriler.info()  
#%%
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#%% 
#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4, metric="minkowski")
knn.fit(X_train, y_train)

knn_predict = knn.predict(X_test)

#Karmasık MatrixeE
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, knn_predict)

print("KNN")
print(cm)


#Figure
n_neighbors = 3
h = .02
X = x[:, :2]
y = y

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])


clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification (k = %i)" % (n_neighbors))
plt.show()


#%% 
#SVC

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(X_train, y_train)

svc_predict = svc.predict(X_test)

cm_svc = confusion_matrix(y_test, svc_predict)
print("SVC")
print(cm_svc)

#%% 
#Naive Bayes

from sklearn.naive_bayes import GaussianNB
nbayes = GaussianNB()
nbayes.fit(X_train, y_train)

nbayes_predict = nbayes.predict(X_test)

cm_nbayes = confusion_matrix(y_test, nbayes_predict)
print("Naive Bayes")
print(cm_nbayes)

#%% 
#Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy')
dtree.fit(X_train, y_train)

dtree_predict = dtree.predict(X_test)

cm_dtree = confusion_matrix(y_test, dtree_predict)
print("Decision Tree")
print(cm_dtree)

#%% 
# Random Forrest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion="entropy")
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)
frc_probo = rfc.predict_proba(X_test)

cm_rfc = confusion_matrix(y_test, rfc_predict)
print("Random Forrest Classifier")
print(cm_rfc)


