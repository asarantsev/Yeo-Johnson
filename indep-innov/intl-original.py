# This is a small version of the simulator with only one asset class:
# international stocks (choose either developed or emerging markets)
# and one factor: volatility, without Yeo-Johnson transform
# We simulate nominal total geometric returns

import pandas as pd
import numpy as np
from statsmodels.api import OLS
from scipy import stats
from matplotlib import pyplot as plt

# read data
np.random.seed(0)
DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
N = len(vol)
# intl = DF['International'].values[43:]
intl = DF['Emerging'].values[61:]
M = len(intl)
lvol = np.log(vol)
total = np.log(1 + intl) # computation of total returns
Nret = total/vol[-M:] # normalization

# fitting simple linear regressions
RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
RegIntl = OLS(Nret, pd.DataFrame({'const' : 1/vol[-M:], 'vol' : 1})).fit()

# and recording coefficients and standard errors
intVol = RegVol.params['const']
slopeVol = RegVol.params['lag']
stdVol = np.std(RegVol.resid)
intIntl = RegIntl.params['const']
slopeIntl = RegIntl.params['vol']
stdIntl = np.std(RegIntl.resid)

# covariance matrix of residuals
covVol = np.linalg.inv(np.array([[N - 1, np.sum(lvol[:-1])], [np.sum(lvol[:-1]), np.sum(np.square(lvol[:-1]))]]))
covIntl = np.linalg.inv(np.array([[M, np.sum(1/vol[-M:])], [np.sum(1/vol[-M:]), np.sum(np.square(1/vol[-M:]))]])) 
print(covVol)
print(covIntl)

# number of simulations
NSIMS = 10000

# frequentist simulation for T years
def classicSim(initVol, T):
    noiseIntl = np.random.normal(0, stdIntl, (T, NSIMS))
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS))
    simRetIntl = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = intVol * np.ones(NSIMS) + slopeVol * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetIntl[t] = intIntl * np.ones(NSIMS) + slopeIntl * simVol[t + 1]  + simVol[t + 1] * noiseIntl[t]
        
    return simRetIntl

# Bayesian simulation with prior upon regression coefficients
def bayesCoeffSim(initVol, T):
    noiseIntl = np.random.normal(0, stdIntl, (T, NSIMS))
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS))
    simRetIntl = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    simCoeffVol = np.random.multivariate_normal([intVol, slopeVol], covVol * stdVol**2, NSIMS)
    simCoeffIntl = np.random.multivariate_normal([slopeIntl, intIntl], covIntl * stdIntl**2, NSIMS)
    
    alpha = simCoeffVol[:, 0]
    beta = simCoeffVol[:, 1]
    theta = simCoeffIntl[:, 0]
    gamma = simCoeffIntl[:, 1]
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = alpha * np.ones(NSIMS) + beta * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetIntl[t] = gamma * np.ones(NSIMS) + theta * simVol[t + 1]  + simVol[t + 1] * noiseIntl[t]
        
    return simRetIntl

# Bayesian simulation with prior upon standard errors
# and upon regression coefficients
def bayesAllSim(initVol, T):
    noiseIntl = np.random.normal(0, stdIntl, (T, NSIMS))
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS))
    simRetIntl = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    simPrecVol = np.random.gamma((N - 1)/2, 2*stdVol**(-2)/(N - 1), NSIMS)
    simStdVol = np.power(simPrecVol, -0.5)
    simCoeffVol = np.tile([intVol, slopeVol], (NSIMS, 1)) + np.random.multivariate_normal([0, 0], covVol, NSIMS) * np.transpose(np.tile(simStdVol, (2, 1)))
    simPrecIntl = np.random.gamma(M/2, 2*stdIntl**(-2)/M, NSIMS)
    simStdIntl = np.power(simPrecIntl, -0.5)
    simCoeffIntl = np.tile([slopeIntl, intIntl], (NSIMS, 1)) + np.random.multivariate_normal([0, 0], covIntl, NSIMS) * np.transpose(np.tile(simStdIntl, (2, 1)))
    
    alpha = simCoeffVol[:, 0]
    beta = simCoeffVol[:, 1]
    theta = simCoeffIntl[:, 0]
    gamma = simCoeffIntl[:, 1]
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = alpha * np.ones(NSIMS) + beta * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetIntl[t] = gamma * np.ones(NSIMS) + theta * simVol[t + 1]  + simVol[t + 1] * noiseIntl[t]
        
    return simRetIntl

T = 30
initVol = 20
model0 = sim(initVol, T)
model1 = bayesCoeffSim(initVol, T)
model2 = bayesAllSim(initVol, T)

for model in [model0, model1, model2]:
    avgModel = np.mean(model, axis = 0)
    print('mean = ', np.mean(avgModel))
    print('std = ', np.std(avgModel))
    print('median = ', np.median(avgModel))
    for percent in [10, 30, 70, 90]:
        print(str(percent) + '% = ', np.percentile(avgModel, percent))
        
    # Now test the withdrawal rule
    wealth = np.zeros((T+1, NSIMS))
    wealth[0] = np.ones(NSIMS)
    for withdrawal in [0.03, 0.04, 0.05]:
        for t in range(T):
            wealth[t+1] = wealth[t] * np.exp(model[t]) - withdrawal * (1.04**t) * np.ones(NSIMS)
         print('survival probability with withdrawal rate ', withdrawal, ' and growth 4% per year')
        print(np.sum(wealth[T] > 0)/NSIMS)
