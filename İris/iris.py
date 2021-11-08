#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#%%
#Veriler
veriler = pd.read_excel('Iris.xls')

x = veriler.iloc[:,0:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken

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
#Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, y_train)

log_predict = log_reg.predict(X_test)

cm_log = confusion_matrix(y_test, log_predict)
print("Logistic Regression")
print(cm_log)

#%% 
#KNN

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski')
KNN.fit(X_train, y_train)

knn_predict = KNN.predict(X_test)

cm_knn = confusion_matrix(y_test, knn_predict)
print("KNN")
print(cm_knn)

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


from sklearn import metrics
print('ROC/AUC')
fpr, tpr, thold = metrics.roc_curve(y_test, frc_probo[:,0],pos_label='e')

print(fpr)
print(tpr)
print(thold)






























