import numpy as np
# Import shikit learn module classes
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler, MinMaxScaler, KernelCenterer

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

from matplotlib.colors import ListedColormap

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

import seaborn as sns

class Classifiers(object):
	"""
		Classifiers class contains multiple classifiers that can be used for predictions.
		It can be used to make predictions
	"""
	def __init__(self):
		"""
			Initalizes the dictionary with most important classifiers. 
			As default or current classifier Kernel Support Vector Classifier is selected.
		"""
		self.random_seed = 0

		self.models = {
			'LogisticRegression' : LogisticRegression(random_state = self.random_seed),
			'RandomForest' : RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=self.random_seed),
			'NaiveBayes' : GaussianNB(),
			'KNN' : KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2),
			'KernelSVC' : SVC(kernel = 'rbf', random_state=self.random_seed),
			'DecisionTree' : DecisionTreeClassifier(criterion='gini', random_state=self.random_seed),
			'AdaBoost' : AdaBoostClassifier(n_estimators=100, random_state=self.random_seed, base_estimator=DecisionTreeClassifier(criterion='entropy', random_state=self.random_seed)),
			'MLP' : MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(5, 2), random_state=self.random_seed)
		}

		self.current = self.models["KernelSVC"]

	def listClassifiers(self):
		return list(self.models.keys())

	def importClassifier(self, name, classifier):
		"""
			This function add new classifier in models dictionary
			or print message if classifier already exists.
		"""
		if name not in self.models.keys():
			self.models[name] = classifier
		else:
			print('Classifier already exists')

	def setClassifier(self, name):
		"""
			Set new current classifier from dictinary of classifiers
		"""
		if name in self.models.keys():
			self.current = self.models[name]
		else:
			print(name + " classifier does not exist")

	def setRanomSeed(self, random_seed):

		self.random_seed = random_seed

	def fit(self, X_train, Y_train):
		"""
			Fit the data with curent classifier
		"""
		self.current.fit(X_train, Y_train)

	def predict(self, X_test):
   		
   		return self.current.predict(X_test)
	
	def LDA(self, X_train, Y_train, X_test, numComponents):
    	
		lda = LinearDiscriminantAnalysis(n_components = numComponents)
		x_train = lda.fit_transform(X_train, Y_train)
		x_test = lda.transform(X_test)

		return (x_train, x_test)


	def holdOutSplit(self, X, Y, test_size, stratify):
		"""
			This function implements hold-out split.X_test
			Return:
				(X_train, X_test, y_train, y_test)
		"""
		return train_test_split( X, Y, test_size=test_size, random_state=self.random_seed, stratify=stratify)

	def scale(self, X_train, X_test):
		"""
			This function performs Standard Scaling on Features
		"""
		sc_X = StandardScaler()
		x_train = sc_X.fit_transform(X_train)
		x_test = sc_X.transform(X_test)

		return (x_train, x_test)

	def minMaxScale(self, X_train, X_test):

		mm_X = MinMaxScaler()
		x_train = mm_X.fit_transform(X_train)
		x_test = mm_X.transform(X_test)

		return (x_train, x_test)


	def CVScore(self, X_train, Y_train, num_splits, shuffle=True):
		"""
			This Function performs cross validation with num_splits
		"""
		folds = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=self.random_seed)
		accuracies = cross_val_score(estimator = self.current, X = X_train, y = Y_train, cv = folds, scoring='f1_micro')
		
		return accuracies

	def GridSearch(self, X_train, Y_train, parameters, folds):
		"""
			This function implements gridSearch.
			Example of parameters:
				parameters = [
 	            	{'C' : [1, 10], 'kernel' : ['rbf'], 'gamma' : np.linspace(0, 0.5, 3).tolist()} #after performing chose another params around optimal value [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
             	]
            Folds parameter can be integer or object like StratifiedKFold
		"""
		grid_search = GridSearchCV(estimator=self.current, param_grid=parameters, scoring='f1_micro', cv=folds) # n_jobs = -1 if you use this on large dataset
		grid_search.fit(X_train, y_train)
		best_accuracy = grid_search.best_score_
		best_parameters = grid_search.best_params_

		return (best_accuracy, best_parameters)

	def plotResults(self, X, Y, title="Classifier", xlabel="LD1", ylabel="LD2"):
		"""
			This function can be used for data vizualization (classes). 
			X must consist exactly 2 arrays(features) and y is predictied class.
			It is usualy combined with Feature Extraction methods like: PCA, LDA, Kernel PCA etc.
		"""
		X_set, y_set = X, Y
		X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
		                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
		             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
		plt.xlim(X1.min(), X1.max())
		plt.ylim(X2.min(), X2.max())
		for i, j in enumerate(np.unique(y_set)):
		    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
		                c = ListedColormap(('red', 'green'))(i), label = j)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.legend()
		plt.show()

	def l1FeatureSelection(self, X, Y):
		
		lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
		model = SelectFromModel(lsvc, prefit=True)
		X_new = model.transform(X)
		return X_new

	def treeBasedFeatureSelection(self, X, Y):
		clf = ExtraTreesClassifier()
		clf = clf.fit(X, Y)
		model = SelectFromModel(clf, prefit=True)
		X_new = model.transform(X)
		return X_new


	def visualizeCorrelation(self, data):
		#ata=X[:300,:].transpose()
		R = np.corrcoef(data)
		pcolor(R)
		colorbar()
		yticks(arange(0,26),range(0,26))
		xticks(arange(0,26),range(0,26))
		show()

		print(R)