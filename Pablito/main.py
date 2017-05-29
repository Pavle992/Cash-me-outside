import numpy as np
from preprocessing import x, y
from Classifiers import Classifiers
# Init Classifiers object
models = Classifiers()

x = np.delete(x, [11, 13, 16, 17, 18, 19, 22, 25], 1)
# x = np.delete(x, [5], 1)
#x = models.l1FeatureSelection(x, y)

# Splitting the dataset into training and test set
X_train, X_test, Y_train, Y_test = models.holdOutSplit(x, y, 0.3, y)

# Feature scaling
X_train, X_test = models.scale(X_train, X_test)
#X_train, X_test = models.minMaxScale(X_train, X_test)

# Applying Linear Discriminat Analysis
#X_train, X_test = models.LDA(X_train, Y_train, X_test, 2)

# Aplying PCA
#X_train, X_test = models.PCA(X_train, X_test, 7)

# OVersampling
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0, k_neighbors=10, kind='svm', m_neighbors=20)
X_train, Y_train = sm.fit_sample(X_train, Y_train)

# Fitting classifier to the Training set
models.setClassifier("NaiveBayes")
models.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = models.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average='binary', pos_label=1)
print(cm)
print(f1)


# Applying k-Fold Cross Validation
accuracy = models.CVScore(x, y, 10)

print(accuracy.mean())
print(accuracy.std())