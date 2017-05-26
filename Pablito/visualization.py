import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualizeCorrelations(df, startIndex, stopIndex):
	sns.set(context="paper", font="monospace")
	g = df.corr()
	plt.figure(figsize=(20,14))
	_ = sns.heatmap(g.iloc[startIndex:stopIndex+1, :], annot =True)
	_.set_xticklabels(labels = df.columns.values,rotation=90)
	_.set_yticklabels(labels = df.columns.values,rotation=0)
	sns.plt.show()

df = pd.read_csv("train_full.csv", sep=';')
df = df.drop('CUST_COD', 1)

visualizeCorrelations(df, 0, 10)