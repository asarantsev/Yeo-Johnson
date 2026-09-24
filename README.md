# Yeo-Johnson
Yeo-Johnson transform of log annual volatility for objective Bayes inference.

bayes-yjx.xlsx for the data 1927-2025

vol-transform.py fits the autoregression of order 1 of transformed logarithmic volatility. We confirm that these innovations are IID Gaussian. This uses the Yeo-Johnson transform, which is similar to the Box-Cox transform but allows for negative values (both originals and images). 

YJX.py contains the inverse Yeo-Johnson transform. We apply this after performing Monte Carlo simulation transformed log volatility to get real volatility. 

Then we have three folders: 

indep-innov for the case when innovations for autoregressions of volatility and for regressions of returns are independent; returns are nominal

nominal and real for the case when the innovations are bivariate Gaussian but dependent; returns are nominal and real, correspondingly

In each folder, we have four files, each has classic (frequentist), Bayesian-lite, and Bayesian-full modes for simulations

usa-original.py: domestic returns with original volatility

usa-transformed.py: domestic returns with transformed volatility

intl-original.py: developed and emerging returns with original volatility

intl-transformed.py: developed and emerging returns with transformed volatility

See also my blog post https://my-finance.org/2026/09/03/yeo-johnson-transform-objective-bayes-inference-for-stock-returns-with-volatility-factor/
