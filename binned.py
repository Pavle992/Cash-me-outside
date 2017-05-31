#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:32:49 2017

@author: korda
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

dataset = pd.read_csv('train_binned.csv', sep=',')

#dataset["BILL_AMT_DEC"] = dataset["BILL_AMT_DEC"].map(lambda x: float(x.replace(',', '')))
dataset.iloc[:,8:20] = dataset.iloc[:,8:20].applymap(lambda x: float(x.replace(',', '')))

y = dataset['DEFAULT PAYMENT JAN']
x = dataset = dataset.drop(['DEFAULT PAYMENT JAN'], 1)




# Encoding categorical variables

labelEncoder = LabelEncoder()
x.iloc[:, 8] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 9] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 10] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 11] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 12] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 13] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 14] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 15] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 16] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 17] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 18] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
x.iloc[:, 19] = labelEncoder.fit_transform(x.iloc[:, 8]).flatten()
#
#
oneHotEncoder = OneHotEncoder(categorical_features=[8,9,10,11,12,13,14,15,16,17,18,19])
X = oneHotEncoder.fit_transform(x.values).toarray()
X = np.delete(X, [0, 10, 20,30,40,50,60,70,80,90], 1)

# Remove Customer ID
#X = np.delete(X, 6, 1)

#x.info()


#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0, stratify=y)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

#X_train = x
#y_train = y

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from imblearn.over_sampling import SMOTE, ADASYN

sm = SMOTE(random_state=42, k_neighbors = 10, kind = 'svm', out_step = 0.5, m_neighbors = 20)
#adasm = ADASYN(random_state=42, n_neighbors = 5)
X_train, y_train = sm.fit_sample(X_train, y_train)

 # Applying Kernel PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

models = {
	'LogisticRegression' : LogisticRegression(random_state = 0), #max 80,98
	'RandomForest' : RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0, n_jobs = -1),
	'NaiveBayes' : GaussianNB(), #0.51
	'KNN' : KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2),
	'KernelSVM' : SVC(kernel = 'rbf', random_state=0),
     'NuSVM' : NuSVC(nu = 0.3,kernel = 'rbf', random_state = 0),
	'DecisionTree' : DecisionTreeClassifier(random_state=0),
	'AdaBoost' : AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=DecisionTreeClassifier(criterion='gini', random_state=0)),
     'MLPClassifier' : MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
     'XGBClassifier' : XGBClassifier(n_estimators = 3000, learning_rate = 0.02, gamma = 0.00000001, subsample = 0.5),
     'ExtraTreesClassifier' : ExtraTreesClassifier(n_estimators = 50, max_depth = None, random_state = 0, n_jobs = -1),
     'IsolationForest' : IsolationForest()
}

classifier = models['MLPClassifier']
#classifier1 = models['MLPClassifier']
#classifier2 = models['RandomForest']
#classifier3 = models['AdaBoost']
#classifier4 = models['XGBClassifier']

#sample_weight = np.array([7 if i == 1 else 1 for i in y_train])
classifier.fit(X_train, y_train)
#eclf1 = VotingClassifier(estimators=[('lr', classifier), ('rf', classifier1), ('gnb', classifier2), ('ada', classifier3), ('xgb', classifier4)], voting='hard')
#eclf1 = eclf1.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#y_pred = eclf1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average = 'binary', pos_label = 1)
print(cm)
print(f1)


