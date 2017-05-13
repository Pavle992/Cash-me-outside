# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

# Importing the dataset
"""
df = pd.read_csv("Project Train Dataset.csv")

df.columns = df.columns.str.replace('"','').str.replace(',',';')
df.iloc[:, 0] = df.iloc[:, 0].str.replace('"','').str.replace(',',';')



df.to_csv('train.csv', index=False)

"""

dataset = pd.read_csv('train.csv', sep=';')

#print(dataset.head())

print(dataset.info())

countSex = dataset['SEX'].value_counts()
dataset['SEX'] = dataset['SEX'].fillna('F', axis = 0)

countEducation = dataset['EDUCATION'].value_counts()
dataset['EDUCATION'] = dataset['EDUCATION'].fillna('university', axis = 0)

countMarriage = dataset['MARRIAGE'].value_counts()
dataset['MARRIAGE'] = dataset['MARRIAGE'].fillna('single', axis = 0)

print(dataset.info())

"""
#print(dataset.describe())
x = dataset.drop('BIRTH_DATE', axis = 1)
x = x.values

labelEncoder = LabelEncoder()
x[:, 2] = labelEncoder.fit_transform(x[:, 2])
#imputer = Imputer(strategy="most_frequent", axis = 0)

#imputer.fit_transform(x)

"""
#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """