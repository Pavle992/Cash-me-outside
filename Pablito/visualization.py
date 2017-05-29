import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualizeCorrelations(df):
	sns.set(context="paper", font="monospace")
	#df = df.iloc[:, :]
	g = df.corr()
	plt.figure(figsize=(16,24))
	_ = sns.heatmap(g.iloc[:, :], annot =True)
	sns.plt.xticks(rotation=60)
	sns.plt.yticks(rotation=60)
	sns.plt.show()

def visualizeDistribution(df, colname):
	pass


df = pd.read_csv("train_full.csv", sep=';')
df = df.drop('CUST_COD', 1)

visualizeCorrelations(df)
# print(df.shape)

# print(df.info())