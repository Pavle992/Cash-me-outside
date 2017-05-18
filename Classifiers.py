# Import shikit learn module classes
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, StratifiedKFold

from matplotlib.colors import ListedColormap

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
		self.models = {
			'LogisticRegression' : LogisticRegression(random_state = 0),
			'RandomForest' : RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
			'NaiveBayes' : GaussianNB(),
			'KNN' : KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2),
			'KernelSVC' : SVC(kernel = 'rbf', random_state=0),
			'DecisionTree' : DecisionTreeClassifier(criterion='gini', random_state=0),
			'AdaBoost' : AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=DecisionTreeClassifier(criterion='entropy', random_state=0))
		
		self.current = models["Kernel SVC"]
	}

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


	def fit(self, X_train, Y_train):
		"""
			Fit the data with curent classifier
		"""
		self.current.fit(X_train, X_train)

    def predict(self, X_test):
    	"""
		Returns data with predictions of current classifier
    	"""

    	return self.current.predict(X_test)

    def LDA(self, X_train, Y_train, X_test, numComponents):   
    	"""
    	This function apply Linear discriminant Analysis as feature extraction method
    	"""
		lda = LinearDiscriminantAnalysis(n_components = numComponents)
		x_train = lda.fit_transform(X_train, Y_train)
		x_test = lda.transform(X_test)

		return (x_train, x_test)


	def hold_out_split(self, X, Y, test_size, random_seed, stratify):
		"""
			This function implements hold-out split.X_test
			Return:
				(X_train, X_test, y_train, y_test)
		"""
		return train_test_split( x, y, test_size=0.25, random_state=0, stratify=y)

	def scale(self, X_train, X_test):
		"""
			This function performs Standard Scaling on Features
		"""
		sc_X = StandardScaler()
		x_train = sc_X.fit_transform(X_train)
		x_test = sc_X.transform(X_test)

		return (xtrain, x_test)

	def CVScore(self, X_train, Y_train, num_splits):
		"""
			This Function performs cross validation with num_splits
		"""
		folds = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=0)
		accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = folds, scoring='f1_micro')
		
		return accuracies

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