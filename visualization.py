#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:13:57 2017

@author: korda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualizeCorrelations(df):
	sns.set(context="paper", font="monospace")
	g = df.corr()
	plt.figure(figsize=(16,24))
	_ = sns.heatmap(g, annot =True)
	sns.plt.xticks(rotation=60)
	sns.plt.yticks(rotation=60)
	sns.plt.show()

