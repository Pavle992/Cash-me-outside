import numpy as np
from preprocessing import x, y
#Building the optimal model using Backward Elimination
#step 1
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((x.shape[0], 1)).astype(int), values=x, axis=1) #adding array of ones as first column
#step 2
# X_opt = np.delete(X, [2, ], 1)
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# # step 2
# X_opt = np.delete(X_opt, 11, 1)
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# # step 2
# X_opt = np.delete(X_opt, [13], 1)
regressor_OLS = sm.OLS(endog=y, exog=X).fit()

print(regressor_OLS.summary())