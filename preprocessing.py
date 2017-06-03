# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from Pablito.visualization import visualizeCorrelations, scatterFeatures, visualizeDistribution
# Importing the dataset
"""
df = pd.read_csv("Project Train Dataset.csv")
df.columns = df.columns.str.replace('"','').str.replace(',',';')
df.iloc[:, 0] = df.iloc[:, 0].str.replace('"','').str.replace(',',';')
df.to_csv('train.csv', index=False)
"""

dataset = pd.read_csv('train.csv', sep=';')

dataset.info()

#print(dataset.head())

#print(dataset.info())
#Prepricessing categorical values

#countSex = dataset['SEX'].value_counts()
dataset['SEX'] = dataset['SEX'].fillna('F', axis = 0)

countEducation = dataset['EDUCATION'].value_counts()
dataset['EDUCATION'] = dataset['EDUCATION'].fillna('university', axis = 0)

#countMarriage = dataset['MARRIAGE'].value_counts()
dataset['MARRIAGE'] = dataset['MARRIAGE'].fillna('single', axis = 0)
dataset.loc[:,'PAY_DEC':'PAY_JUL'] = dataset.loc[:,'PAY_DEC':'PAY_JUL'].replace(to_replace = [-1, -2], value = 0)


#DATA BINNING
#dataset.describe()
billMean = dataset.loc[:,'BILL_AMT_DEC': 'BILL_AMT_JUL'].mean(axis = 1)
payMean = dataset.loc[:,'PAY_AMT_DEC': 'PAY_AMT_JUL'].mean(axis = 1)
#payMax = dataset.loc[:,'PAY_DEC': 'PAY_JUL'].sum(axis = 1)

dataset['BILL_AMT_DEC'] = billMean 
dataset['PAY_AMT_DEC'] = payMean
#dataset['PAY_DEC'] = payMax

dataset = dataset.drop(['BILL_AMT_NOV', 'BILL_AMT_OCT', 'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL'], 1) 
dataset = dataset.drop(['PAY_AMT_NOV', 'PAY_AMT_OCT', 'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL'], 1)
#dataset = dataset.drop(['PAY_DEC', 'PAY_NOV', 'PAY_OCT', 'PAY_SEP', 'PAY_AUG', 'PAY_JUL'], 1) 

#dataset = dataset.drop(['BILL_AMT_DEC', 'PAY_AMT_DEC'], 1)

#print(dataset.info())
dataset['DEFAULT PAYMENT JAN'].value_counts()

#26884 total
# 20940 0
# 5944 1

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

#print(dataset.info())
#visualizeCorrelations(dataset)
#scatterFeatures(dataset, 'LIMIT_BAL', 'DEFAULT PAYMENT JAN')

# Encoding categorical variables

labelEncoder = LabelEncoder()
dataset.iloc[:, 2] = labelEncoder.fit_transform(dataset.iloc[:, 2]).flatten()
dataset.iloc[:, 3] = labelEncoder.fit_transform(dataset.iloc[:, 3]).flatten()
dataset.iloc[:, 4] = labelEncoder.fit_transform(dataset.iloc[:, 4]).flatten()

dataset['SEX'].value_counts()
dataset['EDUCATION'].value_counts()
dataset['MARRIAGE'].value_counts()

oneHotEncoder = OneHotEncoder(categorical_features=[2, 3, 4])
X = oneHotEncoder.fit_transform(dataset.values).toarray()
X = np.delete(X, [0, 4, 6], 1)
# X = np.delete(X, [0, 2, 5], 1)
#print(X[0,:])
# Remove Customer ID
X = np.delete(X, 6, 1)
#X = np.delete(X, 7, 1)
#X = np.delete(X, 11, 1)
#X = np.delete(X, 10, 1)

# Remove BirthDate
X = np.delete(X, 7, 1)

# Remove PAY_AUG
X = np.delete(X, 11, 1)

# Remove PAY_SEP
X = np.delete(X, 10, 1)

# Features
x = X[:, :-1]
y = X[:, -1]

index = np.arange(x.shape[0])
#visualizeCorrelations(pd.DataFrame(X, index=index))
#visualizeDistribution(pd.DataFrame(X, index=index), 4)
# import statsmodels.formula.api as sm
# x = np.append(arr=np.ones((x.shape[0], 1)).astype(int), values=x, axis=1)
# x = np.delete(x, 2, 1)

# regressor_OLS = sm.OLS(endog=y, exog=x).fit()

# print(regressor_OLS.summary())

visualizeDistribution(dataset, 'DEFAULT PAYMENT JAN')



#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

#X_train = x
#y_train = y

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek

sm = SMOTE(random_state=42, k_neighbors = 10, kind = 'svm', out_step = 0.5, m_neighbors = 20)
#adasm = ADASYN(random_state=42, n_neighbors = 5)
ros = RandomOverSampler(random_state=42)
#cc = ClusterCentroids(random_state=42)
#enn = EditedNearestNeighbours(random_state = 42)
#tomek = TomekLinks(random_state = 42)
#smottomek = SMOTETomek(random_state = 42, smote = sm, n_jobs = -1)
X_train, y_train = ros.fit_sample(X_train, y_train)

# #Applying Kernel PCA
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
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, IsolationForest, BaggingClassifier
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier

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
     'XGBClassifier' : XGBClassifier(n_estimators = 100, gamma = 0.00000001, reg_lambda = 100, learning_rate = 0.02, subsample = 0.7),
<<<<<<< HEAD
=======
     'XGBClassifier1' : XGBClassifier(n_estimators = 100, gamma = 0.00000001, reg_lambda = 0.001, learning_rate = 0.02, subsample = 0.3),
     #54.92 reg_lamba = 10 gamma 10 max_depth = 4, n_estimat = 100 l_rate = 0.02 reg_lamd = 10
>>>>>>> 27f8f9ccdeb7496e3404a60c8349d5dc788294c2
     'ExtraTreesClassifier' : ExtraTreesClassifier(n_estimators = 50, max_depth = None, random_state = 0, n_jobs = -1),
     'BaggingClassifier' : BaggingClassifier(XGBClassifier(n_estimators = 100, gamma = 0.00000001, reg_lambda = 100, learning_rate = 0.02, subsample = 0.7), n_estimators = 50, max_samples = 0.5, max_features = 0.5,n_jobs = -1, random_state = 0),
     'BaggingClassifierLogistic' : BaggingClassifier(LogisticRegression(random_state = 0, C = 0.05, penalty = 'l2'), n_estimators = 50, max_samples = 0.5, max_features = 0.5,n_jobs = -1, random_state = 0)
}


classifier = models['BaggingClassifier']
#classifier1 = models['MLPClassifier']
#classifier2 = models['LogisticRegression']
#classifier3 = models['KernelSVM']
#classifier4 = models['NaiveBayes']

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#eclf1 = VotingClassifier(estimators=[('xgb', classifier), ('mlp', classifier1), ('log', classifier2), ('ker', classifier3), ('bais', classifier4)], voting='hard')
#eclf1 = eclf1.fit(X_train, y_train)
#y_pred = eclf1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average = 'binary', pos_label = 1)
print(cm)
print(f1)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score, StratifiedKFold
folds = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
#
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = folds, scoring = 'f1')
#print(accuracies.mean())
#print(accuracies.std())

# Applying Grid Search to find the best model and the best parameters
#from sklearn.model_selection import GridSearchCV
#parameters = [
#	             {'gamma' : [0.00000001],
#                   'subsample' : [0.3],
#                   'reg_lambda' : [0.001],
#                   
#                   
#
#                  } 
#              ]
#
#grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='f1', cv=folds, n_jobs=-1) # n_jobs = -1 if you use this on large dataset
#grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_
#print(best_accuracy)
#print(best_parameters)
#
#results = grid_search.cv_results_

# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('SVC (Training set)')
# plt.xlabel('PC1')
# plt.ylabel('PC1')
# plt.legend()
# plt.show()

# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('SVC (Test set)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()
# plt.show()