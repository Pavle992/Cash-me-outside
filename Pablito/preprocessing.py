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

dataset = pd.read_csv('../train.csv', sep=';')

# Prepricessing categorical values
countSex = dataset['SEX'].value_counts()
dataset['SEX'] = dataset['SEX'].fillna('F', axis = 0)

countEducation = dataset['EDUCATION'].value_counts()
dataset['EDUCATION'] = dataset['EDUCATION'].fillna('university', axis = 0)

countMarriage = dataset['MARRIAGE'].value_counts()
dataset['MARRIAGE'] = dataset['MARRIAGE'].fillna('single', axis = 0)

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

xx = imputer.fit_transform(dataset.iloc[:, 5])
dataset.iloc[:, 5] = xx.flatten()

#dataset.to_csv('train_full.csv', sep=';', index=False)

# print(dataset.head())
# print(dataset.info())
# print(dataset.describe())
# print(dataset[:])

# # Encoding categorical variables

labelEncoder = LabelEncoder()
dataset.iloc[:, 2] = labelEncoder.fit_transform(dataset.iloc[:, 2]).flatten()
dataset.iloc[:, 3] = labelEncoder.fit_transform(dataset.iloc[:, 3]).flatten()
dataset.iloc[:, 4] = labelEncoder.fit_transform(dataset.iloc[:, 4]).flatten()

oneHotEncoder = OneHotEncoder(categorical_features=[2, 3, 4])
X = oneHotEncoder.fit_transform(dataset.values).toarray()
X = np.delete(X, [0, 2, 5], 1)

# Remove Customer ID
X = np.delete(X, 6, 1)
# Replace all negative with zero value
X = np.where(X<0, 0, X)
# Features
x = X[:, :-1]
y = X[:, -1]


#print(dataset.columns)

""""
Note: Columns in Numpy nDarray

Columns:
	0) SEX
  1-3) Education
  4-5) Marrage
    6) Limit Balance
    7) Birth Date
 8-13) PAY_DEC - PAY_JULY
14-19) BILL_AMOUNT_DEC - BILL_AMOUNT_JULY
20-25) PAY_AMOUNT_DEC - PAY_AMOUNT_JULY
   26) DEFAULT_PAYMENT		
"""

# Percentage for PAY_AMOUNT

limit_balance = x[:, 6]
pay_amount = x[:, 20:26].T

new_data = np.divide(pay_amount, limit_balance)
x[:, 20:26] = new_data.T

# # Add new column - Mean of PAY_AMOUNTS
# new_col = x[:, 20:26].mean(axis=1)
# new_col = new_col.reshape(-1, 1)

# x = np.append(x, new_col, 1)
# x = np.delete(x, [20, 21, 22, 23, 24, 25], 1)

# Percentage for BILL_AMOUNT

bill_amount = x[:, 14:20].T

new_data = np.divide(bill_amount, limit_balance)
x[:, 14:20] = new_data.T

# # Add new column - Mean of BILL_AMOUNT
# new_col = x[:, 14:20].mean(axis=1)
# new_col = new_col.reshape(-1, 1)

# x = np.append(x, new_col, 1)
# x = np.delete(x, [14, 15, 16, 17, 18, 19], 1)

# # PAY_DEC - PAY_JUL

# pay = x[:, 8:14]

# new_col = x[:, 8:14].sum(axis=1)
# new_col = new_col.reshape(-1, 1)

# x = np.append(x, new_col, 1)
# x = np.delete(x, [8, 9, 10, 11, 12, 13], 1)
