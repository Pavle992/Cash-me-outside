import numpy as np
from preprocessing
#Building the optimal model using Backward Elimination
#step 1
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1) #adding array of ones as first column
#step 2
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())