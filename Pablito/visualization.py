import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

def visualizeCorrelations(df):
	sns.set(context="paper", font="monospace")
	#df = df.iloc[:, :]
	g = df.corr()
	plt.figure(figsize=(16,24))
	_ = sns.heatmap(g.iloc[:, :], annot =True)
	sns.plt.xticks(rotation=60)
	sns.plt.yticks(rotation=60)
	sns.plt.show()

def visualizeDistribution(df, colName):
	#print(df.columns.values)
	plt.hist(df.loc[:, colName])  # plt.hist passes it's arguments to np.histogram
	plt.title("Histogram of " + colName)
	plt.show()

def scatterFeatures(df, f1_name, f2_name):
	plt.scatter(df[f1_name], df[f2_name])
	print(df[f2_name].min(), df[f2_name].max())

	# plt.ylim([df[f2_name].min(), df[f2_name].max()])
	# plt.xlim([df[f1_name].min(), df[f1_name].max()])
	plt.xlabel(f1_name)
	plt.ylabel(f2_name)
	plt.show()


# df = pd.read_csv("train_full.csv", sep=';')
# df = df.drop('CUST_COD', 1)

"""
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

#visualizeCorrelations(df)
# print(df.shape)

# print(df.info())
# visualizeDistribution(df, 'DEFAULT PAYMENT JAN')
# visualizeDistribution(df, 'LIMIT_BAL')
# visualizeDistribution(df, 'BILL_AMT_NOV')