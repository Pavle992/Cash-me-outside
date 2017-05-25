# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import Orange
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

# Importing the dataset
"""
df = pd.read_csv("Project Train Dataset.csv")

df.columns = df.columns.str.replace('"','').str.replace(',',';')
df.iloc[:, 0] = df.iloc[:, 0].str.replace('"','').str.replace(',',';')



df.to_csv('train.csv', index=False)

"""
dataset = pd.read_csv('train.csv', sep=';')

#print(dataset.head())

#print(dataset.info())

#Prepricessing categorical values

countSex = dataset['SEX'].value_counts()
dataset['SEX'] = dataset['SEX'].fillna('F', axis = 0)

countEducation = dataset['EDUCATION'].value_counts()
dataset['EDUCATION'] = dataset['EDUCATION'].fillna('university', axis = 0)


countMarriage = dataset['MARRIAGE'].value_counts()
dataset['MARRIAGE'] = dataset['MARRIAGE'].fillna('single', axis = 0)



# billMean = dataset.loc[:,'BILL_AMT_DEC': 'BILL_AMT_JUL'].sum(axis = 1)
# payMean = dataset.loc[:,'PAY_AMT_DEC': 'PAY_AMT_JUL'].sum(axis = 1)

# dataset['BILL_AMT_DEC'] = billMean 
# dataset['PAY_AMT_DEC'] = payMean
#print(dataset.info())

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
X=np.where(X<0,0,X)
# X1=np.putmask(X,X<0,0)
# print(X1)
# print(X)
# X=np.mean(X,axis=1)
# Features
x = X[:, :-1]
y = X[:, -1]



# bill_amount = x[:, 14:20].T

# new_data = np.divide(bill_amount, limit_balance)
# x[:, 14:20] = new_data.T

# new_col = x[:, 14:20].mean(axis=1)
# new_col = new_col.reshape(-1, 1)

# x = np.append(x, new_col, 1)
# x = np.delete(x, [14, 15, 16, 17, 18, 19], 1)

# limit_balance = x[:, 6]
# pay_amount = x[:, 20:26].T

# new_data = np.divide(pay_amount, limit_balance)
# x[:, 20:26] = new_data.T

# x = np.append(x, new_col, 1)
# x = np.delete(x, [20, 21, 22, 23, 24, 25], 1)

# print(x)
# x=Orange.data.Table('x')
# disc=Orange.preprocess.Discretize()
# disc.method=Orange.preprocess.discretize.EqualFreq(n=3)
# x=disc(x)

# print(countSex)
# lsvc=LinearSVC(C=0.01, penalty='l1', dual=False).fit(x,y)
# model=SelectFromModel(lsvc, prefit=True)
# x=model.transform(x)
 #Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=0, stratify=y)
# print(X_train)
# #Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# # Applying Kernel PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components = 2)
# X_train = lda.fit_transform(X_train, y_train)
# X_test = lda.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
models = {
	'LogisticRegression' : LogisticRegression(random_state = 0),
	'RandomForest' : RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
	'NaiveBayes' : GaussianNB(),
	'KNN' : KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2),
	'KernelSVM' : SVC(kernel = 'rbf', random_state=0, gamma=0.14),
	'DecisionTree' : DecisionTreeClassifier(criterion='entropy', random_state=0),
	'AdaBoost' : AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=DecisionTreeClassifier(criterion='gini', random_state=0)),
	'MLP' : MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes=(5, 2), random_state=0),
	'LinearSVC': LinearSVC(C=1, class_weight=None, dual=False, fit_intercept=True),
	'VS': OneVsRestClassifier(SVC(kernel='rbf', random_state=0, gamma=0.14)),
	

}

classifier = models['MLP']
# classifier = models['LogisticRegression']
# classifier1 = models['MLP']
# classifier2 = models['NaiveBayes']
# classifier3 = models['KernelSVM']
# classifier4 = models['KNN']

# eclf1 = VotingClassifier(estimators=[('lr', classifier), ('rf', classifier1), ('gnb', classifier2), ('ada', classifier3)], voting='hard')
# eclf1 = eclf1.fit(X_train, y_train)
# y_pred = eclf1.predict(X_test)


# estimator.get_params()

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')
print(cm)
print(f1)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
folds= StratifiedKFold(shuffle=True, n_splits=10, random_state=0)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = folds)
# acci=accuracy_score(y_test,y_pred)
# print(acci)
print(accuracies.mean())
print(accuracies.std())

# Applying Grid Search to find the best model and the best parameters
# from sklearn.model_selection import GridSearchCV
#parameters = [
#	             {'C' : [1, 10, 50], 'kernel' : ['rbf'], 'gamma' : np.linspace(0, 0.5, 4).tolist()} #after performing chose another params around optimal value [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
 #             ]

#grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='f1_micro', cv=folds) # n_jobs = -1 if you use this on large dataset
#grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_
#print(best_accuracy)
#print(best_parameters)

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


