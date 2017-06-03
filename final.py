#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:19:50 2017

@author: korda
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from visualization import visualizeCorrelations

import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)

dataset = pd.read_csv('train.csv', sep=';')


#Preprocessing categorical values

dataset['SEX'] = dataset['SEX'].fillna('F', axis = 0)
dataset['EDUCATION'] = dataset['EDUCATION'].fillna('university', axis = 0)
dataset['MARRIAGE'] = dataset['MARRIAGE'].fillna('single', axis = 0)

#Replace negative values in PAY
dataset.loc[:,'PAY_DEC':'PAY_JUL'] = dataset.loc[:,'PAY_DEC':'PAY_JUL'].replace(to_replace = [-1, -2], value = 0)

visualizeCorrelations(dataset)

#Mean on BILL_AMT and PAY_AMT
billMean = dataset.loc[:,'BILL_AMT_DEC': 'BILL_AMT_JUL'].mean(axis = 1)
payMean = dataset.loc[:,'PAY_AMT_DEC': 'PAY_AMT_JUL'].mean(axis = 1)
dataset['BILL_AMT_DEC'] = billMean 
dataset['PAY_AMT_DEC'] = payMean

dataset = dataset.drop(['BILL_AMT_NOV', 'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL'], 1) 
dataset = dataset.drop(['PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL'], 1)


# Age preprocessing
from datetime import date, datetime

def calculate_age(born):
	if isinstance(born, float):
		return born
	born = datetime.strptime(born, "%d/%m/%Y")
	today = date.today()
	return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

dataset["BIRTH_DATE"] = dataset["BIRTH_DATE"].map(lambda x: calculate_age(x))

imputer = Imputer(missing_values = np.nan, strategy="median", axis = 1)

dataset.iloc[:, 5] = imputer.fit_transform(dataset.iloc[:, 5]).flatten()

# Encoding categorical variables
labelEncoder = LabelEncoder()
dataset.iloc[:, 2] = labelEncoder.fit_transform(dataset.iloc[:, 2]).flatten()
dataset.iloc[:, 3] = labelEncoder.fit_transform(dataset.iloc[:, 3]).flatten()
dataset.iloc[:, 4] = labelEncoder.fit_transform(dataset.iloc[:, 4]).flatten()

oneHotEncoder = OneHotEncoder(categorical_features=[2, 3, 4])
X = oneHotEncoder.fit_transform(dataset.values).toarray()
X = np.delete(X, [0, 2, 6], 1)

# Remove Customer ID
X = np.delete(X, 6, 1)

# Features and Labels
X_train = X[:, :-1]
y_train = X[:, -1]


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

#Oversampling the train set
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek

#sm = SMOTE(random_state=42, k_neighbors = 10, kind = 'svm', out_step = 0.5, m_neighbors = 20)
#adasm = ADASYN(random_state=42, n_neighbors = 5)
ros = RandomOverSampler(random_state = 42)
#cc = ClusterCentroids(random_state=42)
#enn = EditedNearestNeighbours(random_state = 42)
#tomek = TomekLinks(random_state = 42)
#smottomek = SMOTETomek(random_state = 42, smote = sm, n_jobs = -1)
X_train, y_train = ros.fit_sample(X_train, y_train)

# Fitting classifier to the Training set
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, IsolationForest, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


models = {
	'LogisticRegression' : LogisticRegression(random_state = 0, C = 0.05, penalty = 'l2'), #max 54,35 penalty = l2 c = 0.05
	'RandomForest' : RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0, n_jobs = -1),
	'NaiveBayes' : GaussianNB(), #0.51
	'KNN' : KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2),
	'KernelSVM' : SVC(kernel = 'rbf', random_state=0),
     'NuSVM' : NuSVC(nu = 0.3,kernel = 'rbf', random_state = 0),
	'DecisionTree' : DecisionTreeClassifier(random_state=0),
	'AdaBoost' : AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=DecisionTreeClassifier(criterion='gini', random_state=0)),
     'MLPClassifier' : MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1),
     'XGBClassifier' : XGBClassifier(n_estimators = 100, max_depth = 9, learning_rate = 0.02, gamma = 0.0000001, reg_lambda = 0.00001, subsample = 0.5),
     #54.92 reg_lamba = 10 gamma 10 max_depth = 4, n_estimat = 100 l_rate = 0.02 reg_lamd = 10
     'ExtraTreesClassifier' : ExtraTreesClassifier(n_estimators = 50, max_depth = None, random_state = 0, n_jobs = -1),
     'IsolationForest' : IsolationForest(),
     'BaggingClassifier' : BaggingClassifier(XGBClassifier(n_estimators = 100, gamma = 0.00000001, reg_lambda = 100, learning_rate = 0.02, subsample = 0.7), n_estimators = 50, max_samples = 0.5, max_features = 0.5,n_jobs = -1, random_state = 0)
}

classifier = models['BaggingClassifier']
#classifier1 = models['MLPClassifier']
#classifier2 = models['LogisticRegression']
#classifier3 = models['KernelSVM']
#classifier4 = models['NaiveBayes']

classifier.fit(X_train, y_train)
#eclf1 = VotingClassifier(estimators=[('xgb', classifier), ('mlp', classifier1), ('log', classifier2), ('ker', classifier3), ('bais', classifier4)], voting='hard')
#eclf1 = eclf1.fit(X_train, y_train)

#y_pred = classifier.predict(X_test)
#y_pred = eclf1.predict(X_test)

test_dataset = pd.read_csv('Project Test Dataset.csv', sep=';')

test_dataset['SEX'] = test_dataset['SEX'].fillna('F', axis = 0)
test_dataset['EDUCATION'] = test_dataset['EDUCATION'].fillna('university', axis = 0)
test_dataset['MARRIAGE'] = test_dataset['MARRIAGE'].fillna('single', axis = 0)

#Replace negative values in PAY
test_dataset.loc[:,'PAY_DEC':'PAY_JUL'] = test_dataset.loc[:,'PAY_DEC':'PAY_JUL'].replace(to_replace = [-1, -2], value = 0)

#Mean on BILL_AMT and PAY_AMT
billMean = test_dataset.loc[:,'BILL_AMT_DEC': 'BILL_AMT_JUL'].mean(axis = 1)
payMean = test_dataset.loc[:,'PAY_AMT_DEC': 'PAY_AMT_JUL'].mean(axis = 1)
test_dataset['BILL_AMT_DEC'] = billMean 
test_dataset['PAY_AMT_DEC'] = payMean

test_dataset = test_dataset.drop(['BILL_AMT_NOV', 'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL'], 1) 
test_dataset = test_dataset.drop(['PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL'], 1)


# Age preprocessing
def calculate_age_test(born):

    if isinstance(born, float):
        return born
    if len(born) < 19:
        born = datetime.strptime(born, "%d/%m/%Y")
    else:
        born = datetime.strptime(born[:10], "%Y-%m-%d")
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

#a = test_dataset[pd.notnull(test_dataset["BIRTH_DATE"])]

#test_dataset["BIRTH_DATE"] = test_dataset[pd.notnull(test_dataset["BIRTH_DATE"])].map(lambda x: x[:11])
test_dataset["BIRTH_DATE"] = test_dataset["BIRTH_DATE"].map(lambda x: calculate_age_test(x))

imputer = Imputer(missing_values = np.nan, strategy="median", axis = 1)

test_dataset.iloc[:, 5] = imputer.fit_transform(test_dataset.iloc[:, 5]).flatten()

# Encoding categorical variables
labelEncoder = LabelEncoder()
test_dataset.iloc[:, 2] = labelEncoder.fit_transform(test_dataset.iloc[:, 2]).flatten()
test_dataset.iloc[:, 3] = labelEncoder.fit_transform(test_dataset.iloc[:, 3]).flatten()
test_dataset.iloc[:, 4] = labelEncoder.fit_transform(test_dataset.iloc[:, 4]).flatten()

test_dataset = test_dataset.drop(['DEFAULT PAYMENT JAN'],1)

oneHotEncoder = OneHotEncoder(categorical_features=[2, 3, 4])
X_test = oneHotEncoder.fit_transform(test_dataset.values).toarray()
X_test = np.delete(X_test, [0, 2, 6], 1)

# Remove Customer ID
X_test = np.delete(X_test, 6, 1)

X_test = sc_X.fit_transform(X_test)

y_pred = classifier.predict(X_test)

print(y_pred.sum())

np.savetxt('test.csv', y_pred, fmt = '%d', header = 'DEFAULT PAYMENT JAN')



