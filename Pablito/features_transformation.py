import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

dataset = pd.read_csv('train_full.csv', sep=';')

#print(dataset.head())

# Encoding categorical variables

labelEncoder = LabelEncoder()
dataset.iloc[:, 2] = labelEncoder.fit_transform(dataset.iloc[:, 2]).flatten()
dataset.iloc[:, 3] = labelEncoder.fit_transform(dataset.iloc[:, 3]).flatten()
dataset.iloc[:, 4] = labelEncoder.fit_transform(dataset.iloc[:, 4]).flatten()

oneHotEncoder = OneHotEncoder(categorical_features=[2, 3, 4])
X = oneHotEncoder.fit_transform(dataset.values).toarray()
X = np.delete(X, [0, 4, 7], 1)

# Remove Customer ID
X = np.delete(X, 6, 1)
# Replace all negative with zero value
#X = np.where(X<0, 0, X)
# Features
x = X[:, :-1]
y = X[:, -1]

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

print(x[:, 6])
print("###########################3")
print(x[:, 20:26])
print("###########################3")
limit_balance = x[:, 6]
pay_amount = x[:, 20:26].T

new_data = np.divide(pay_amount, limit_balance)
x[:, 20:26] = new_data.T
print("###########################3")
print(x[:5, 20:26])

# Percentage for BILL_AMOUNT

limit_balance = x[:, 6]
bill_amount = x[:, 14:20].T

new_data = np.divide(bill_amount, limit_balance)
x[:, 14:20] = new_data.T



