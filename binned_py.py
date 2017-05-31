#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:58:59 2017

@author: korda
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

dataset = pd.read_csv('train.csv', sep=';')


y = dataset['DEFAULT PAYMENT JAN']
#x = dataset.drop(['DEFAULT PAYMENT JAN', 'CUST_COD'], 1)

dataset = dataset.drop(['DEFAULT PAYMENT JAN', 'CUST_COD'], 1)
dataset['SEX'] = dataset['SEX'].fillna('F', axis = 0)
dataset['EDUCATION'] = dataset['EDUCATION'].fillna('university', axis = 0)
dataset['MARRIAGE'] = dataset['MARRIAGE'].fillna('single', axis = 0)
dataset.loc[:,'PAY_DEC':'PAY_JUL'] = dataset.loc[:,'PAY_DEC':'PAY_JUL'].replace(to_replace = [-1, -2], value = 0)

billMean = dataset.loc[:,'BILL_AMT_DEC': 'BILL_AMT_JUL'].mean(axis = 1)
payMean = dataset.loc[:,'PAY_AMT_DEC': 'PAY_AMT_JUL'].mean(axis = 1)
##
dataset['BILL_AMT_DEC'] = billMean 
dataset['PAY_AMT_DEC'] = payMean
#
dataset = dataset.drop(['BILL_AMT_NOV', 'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL'], 1) 
dataset = dataset.drop(['PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL'], 1)
#dataset = dataset.drop(['PAY_DEC', 'PAY_NOV', 'PAY_OCT', 'PAY_SEP', 'PAY_AUG', 'PAY_JUL'], 1) 
#dataset = dataset.drop(['BILL_AMT_DEC', 'PAY_AMT_DEC'], 1)

# Birthdate to date preprocessing
from datetime import date, datetime

def calculate_age(born):
	if isinstance(born, float):
		return born
	born = datetime.strptime(born, "%d/%m/%Y")
	today = date.today()
	return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

dataset["BIRTH_DATE"] = dataset["BIRTH_DATE"].map(lambda x: calculate_age(x))

imputer = Imputer(missing_values = np.nan, strategy="median", axis = 1)

xx = imputer.fit_transform(dataset.iloc[:, 4])
dataset.iloc[:, 4] = xx.flatten()

dataset['PAY_DEC'].value_counts()

lower = dataset.iloc[:,4].min()
upper = dataset.iloc[:,4].max()
bins = np.linspace(lower, upper, 4)
bin_names = ['Bin1', 'Bin2', 'Bin3']

categories = pd.cut(dataset.iloc[:,4], bins, labels=bin_names, include_lowest = True)
dataset.iloc[:,4] = categories

for i in range(5,11):
    
    lower = dataset.iloc[:,i].min()
    upper = dataset.iloc[:,i].max()
    bins = np.linspace(lower, upper, 5)
    
    bin_names = ['Bin1', 'Bin2', 'Bin3', 'Bin4']

    categories = pd.cut(dataset.iloc[:,i], bins, labels=bin_names, include_lowest = True)
    dataset.iloc[:,i] = categories
    
#x['BILL_AMT_DEC'].value_counts()
#x['BILL_AMT_NOV'].value_counts()
#x['BILL_AMT_OCT'].value_counts()
#x['BILL_AMT_SEP'].value_counts()
#x['BILL_AMT_AUG'].value_counts()
#x['BILL_AMT_JUL'].value_counts()
#
dataset['PAY_DEC'].value_counts()
dataset['PAY_NOV'].value_counts()
dataset['PAY_OCT'].value_counts()
dataset['PAY_SEP'].value_counts()
dataset['PAY_AUG'].value_counts()
dataset['PAY_JUL'].value_counts()

    

# Encoding categorical variables

labelEncoder = LabelEncoder()
dataset.iloc[:, 4] = labelEncoder.fit_transform(dataset.iloc[:, 4]).flatten()

dataset.iloc[:, 5] = labelEncoder.fit_transform(dataset.iloc[:, 5]).flatten()
dataset.iloc[:, 6] = labelEncoder.fit_transform(dataset.iloc[:, 6]).flatten()
dataset.iloc[:, 7] = labelEncoder.fit_transform(dataset.iloc[:, 7]).flatten()
dataset.iloc[:, 8] = labelEncoder.fit_transform(dataset.iloc[:, 8]).flatten()
dataset.iloc[:, 9] = labelEncoder.fit_transform(dataset.iloc[:, 9]).flatten()
dataset.iloc[:, 10] = labelEncoder.fit_transform(dataset.iloc[:, 10]).flatten()


dataset.iloc[:, 1] = labelEncoder.fit_transform(dataset.iloc[:, 1]).flatten()
dataset.iloc[:, 2] = labelEncoder.fit_transform(dataset.iloc[:, 2]).flatten()
dataset.iloc[:, 3] = labelEncoder.fit_transform(dataset.iloc[:, 3]).flatten()

#
#
#x['SEX'].value_counts()
#x['EDUCATION'].value_counts()
#x['MARRIAGE'].value_counts()


oneHotEncoder = OneHotEncoder(categorical_features=[1,2,3,4])
dataset = oneHotEncoder.fit_transform(dataset.values).toarray()
#x = np.delete(x, [0, 2, 6, 9, 14, 19, 24, 29, 34, 39 , 44, 49, 54, 59], 1)
dataset = np.delete(dataset, [0, 2, 6, 9], 1)

x = dataset



# Remove Customer ID
#X = np.delete(X, 6, 1)

#x.info()


#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=0, stratify=y)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

#X_train = x
#y_train = y

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from imblearn.over_sampling import SMOTE, ADASYN

sm = SMOTE(random_state=42, k_neighbors = 10, kind = 'svm', out_step = 0.5, m_neighbors = 20)

X_train, y_train = sm.fit_sample(X_train, y_train)

 # Applying Kernel PCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#lda = LinearDiscriminantAnalysis(n_components = 2)
#X_train = lda.fit_transform(X_train, y_train)
#X_test = lda.transform(X_test)

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
	'RandomForest' : RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0, n_jobs = -1),
	'NaiveBayes' : GaussianNB(), #0.51
	'KNN' : KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2),
	'KernelSVM' : SVC(kernel = 'rbf', random_state=0),
     'NuSVM' : NuSVC(nu = 0.3,kernel = 'rbf', random_state = 0),
	'DecisionTree' : DecisionTreeClassifier(random_state=0),
	'AdaBoost' : AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=DecisionTreeClassifier(criterion='gini', random_state=0)),
     'MLPClassifier' : MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
     'XGBClassifier' : XGBClassifier(n_estimators = 100, max_depth = 4, learning_rate = 0.02, gamma = 0.1, subsample = 0.5),
     'ExtraTreesClassifier' : ExtraTreesClassifier(n_estimators = 50, max_depth = None, random_state = 0, n_jobs = -1),
     'IsolationForest' : IsolationForest()
}

classifier = models['XGBClassifier']
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


