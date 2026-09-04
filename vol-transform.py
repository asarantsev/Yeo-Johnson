# Apply the Yeo-Johnson transform to log volatility
# and fit autoregression of order 1
# to make regression residuals IID Gaussian
# For original log volatility, autoregression produces
# borderline cases of Gaussianity for residuals
# We must apply first Yeo-Johnson transform then run autoregression
# in order to do Bayesian inference for Gaussian likelihood

import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from YJX import YJinv # inverse Yeo-Johnson transform

# reading the data file
DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
lvol = np.log(vol)

# This is the quantile-quantile plot for log volatility
qqplot(lvol, line = 's')
plt.title('Original Log Volatility vs Gaussian Law')
plt.show()

# apply Yeo-Johnson transform to log volatility
nvol = scipy.stats.yeojohnson(lvol)[0]
la = scipy.stats.yeojohnson(lvol)[1]
print('lambda = ', la)

# This is the quantile-quantile plot for transformed log volatility
qqplot(nvol, line = 's')
plt.title('Transformed Log Volatility vs Gaussian Law')
plt.show()

# apply the inverse transform to test this function
# We have 1-1 match so the inverse function is correct
testvol = YJinv(nvol, la)
plt.plot(testvol, lvol, 'o')
plt.show()

# fit autoregression for transformed volatility
RegVol = OLS(np.diff(nvol), pd.DataFrame({'const' : 1, 'lag' : nvol[:-1]})).fit()
print(RegVol.summary())
volresid = RegVol.resid

# test residuals for Gaussianity
qqplot(volresid, line = 's')
plt.title('AR(1) Residuals for Transformed Log Volatility')
plt.show()
print('Shapiro-Wilk test p = ', scipy.stats.shapiro(volresid)[1])
print('Jarque-Bera test p = ', scipy.stats.jarque_bera(volresid)[1])

# ACF plots for original and absolute values of residuals
# Show that these are IID
plot_acf(volresid, zero = False)
plt.title('Original values of autoregression residuals')
plt.show()
plot_acf(abs(volresid), zero = False)
plt.title('Absolute values of autoregression residuals')
plt.show()