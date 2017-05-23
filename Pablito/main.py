from preprocessing import x, y
from Classifiers import Classifiers
# Init Classifiers object
models = Classifiers()

#x = np.delete(x, [11, 13, 16, 17, 18, 19, 22, 25], 1)

x = models.l1FeatureSelection(x, y)

# Splitting the dataset into training and test set
X_train, X_test, Y_train, Y_test = models.holdOutSplit(x, y, 0.25, y)

# Feature scaling
X_train, X_test = models.scale(X_train, X_test)
#X_train, X_test = models.minMaxScale(X_train, X_test)

# Applying Linear Discriminat Analysis
#X_train, X_test = models.LDA(X_train, Y_train, X_test, 2)

# Fitting classifier to the Training set
models.setClassifier("KernelSVC")
models.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = models.predict(X_test)

# Applying k-Fold Cross Validation
accuracy = models.CVScore(X_train, Y_train, 10)

print(accuracy.mean())
print(accuracy.std())