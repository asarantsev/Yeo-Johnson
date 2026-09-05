# Yeo-Johnson
Yeo-Johnson transform of log annual volatility for objective Bayes inference.

full-data.xlsx for the data 1927-2025

vol-transform.py fits the autoregression of order 1 of transformed logarithmic volatility. We confirm that these innovations are IID Gaussian. This uses the Yeo-Johnson transform, which is similar to the Box-Cox transform but allows for negative values (both originals and images). 

YJX.py contains the inverse Yeo-Johnson transform. We apply this after performing Monte Carlo simulation transformed log volatility to get real volatility. 

The following four files each have classic (frequentist), Bayesian-lite, and Bayesian-full modes for simulations. 

usa-original.py: domestic returns with original volatility.

usa-transformed.py: domestic returns with transformed volatility.

intl-original.py: developed and emerging returns with original volatility.

intl-transformed.py: developed and emerging returns with transformed volatility.

https://my-finance.org/2026/09/03/yeo-johnson-transform-objective-bayes-inference-for-stock-returns-with-volatility-factor/

Update: Apologies, I absolutely forgot that series of the two innovations also need to be correlated. All this was done for two independent series of residuals. But we still benefited from it, since we made this mistake for both original and transformed data; and for classic vs Bayesian. We need to include covariance matrix of innovations, for both classic and Bayesian versions. 
