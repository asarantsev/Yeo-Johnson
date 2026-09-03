# This is a small version of the simulator with only one asset class: USA stocks
# and one factor: volatility, without Yeo-Johnson transform
# We simulate nominal total geometric returns

import pandas as pd
import numpy as np
from statsmodels.api import OLS
from scipy import stats

# read data
np.random.seed(0)
DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
N = len(vol)
price = DF['Price'].values
dividend = DF['Dividends'].values[1:]
lvol = np.log(vol)

# computation of total returns
total = np.array([np.log(price[k+1] + dividend[k]) - np.log(price[k]) for k in range(N)])

# normalization
nUSAret = total/vol

# fitting simple linear regressions
RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
RegUSA = OLS(nUSAret, pd.DataFrame({'const' : 1/vol, 'vol' : 1})).fit()

# and recording coefficients and standard errors
intVol = RegVol.params['const']
slopeVol = RegVol.params['lag']
stdVol = np.std(RegVol.resid)
intUSA = RegUSA.params['const']
slopeUSA = RegUSA.params['vol']
stdUSA = np.std(RegUSA.resid)

# covariance matrix of residuals
covVol = np.linalg.inv(np.array([[N - 1, np.sum(lvol[:-1])], [np.sum(lvol[:-1]), np.sum(np.square(lvol[:-1]))]]))
covUSA = np.linalg.inv(np.array([[N, np.sum(1/vol)], [np.sum(1/vol), np.sum(np.square(1/vol))]])) 

# number of simulations
NSIMS = 10000

# frequentist simulation for T years
def classicSim(initVol, T):
    noiseUSA = np.random.normal(0, stdUSA, (T, NSIMS)) # generate Gaussian noise for returns
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS)) # generate Gaussian noise for volatility
    simRetUSA = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = intVol * np.ones(NSIMS) + slopeVol * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetUSA[t] = intUSA * np.ones(NSIMS) + slopeUSA * simVol[t + 1]  + simVol[t + 1] * noiseUSA[t]
        
    return simRetUSA

# Bayesian simulation with prior upon regression coefficients
def bayesCoeffSim(initVol, T):
    noiseUSA = np.random.normal(0, stdUSA, (T, NSIMS)) # generate Gaussian noise for returns
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS)) # generate Gaussian noise for volatility
    simRetUSA = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    # simulation of regression coefficients using posteriors
    simCoeffVol = np.random.multivariate_normal([intVol, slopeVol], covVol * stdVol**2, NSIMS)
    simCoeffUSA = np.random.multivariate_normal([slopeUSA, intUSA], covUSA * stdUSA**2, NSIMS)
    
    # return random regression coefficients
    alpha = simCoeffVol[:, 0]
    beta = simCoeffVol[:, 1]
    theta = simCoeffUSA[:, 0]
    gamma = simCoeffUSA[:, 1]
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = alpha * np.ones(NSIMS) + beta * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetUSA[t] = gamma * np.ones(NSIMS) + theta * simVol[t + 1]  + simVol[t + 1] * noiseUSA[t]
        
    return simRetUSA

# Bayesian simulation with prior upon standard errors
# and upon regression coefficients
def bayesAllSim(initVol, T):
    noiseUSA = np.random.normal(0, stdUSA, (T, NSIMS)) # generate Gaussian noise for returns
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS)) # generate Gaussian noise for volatility
    simRetUSA = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    # simulation of volatility
    # simulation of precision and standard errors using Bayesian posterior
    simPrecVol = np.random.gamma((N - 1)/2, 2*stdVol**(-2)/(N - 1), NSIMS)
    simStdVol = np.power(simPrecVol, -0.5)
    
    # simulation of regression coefficients using Bayesian posterior
    errVol = np.random.multivariate_normal([0,0],covVol,NSIMS) * np.transpose(np.tile(simStdVol,(2,1)))
    simCoeffVol = np.tile([intVol, slopeVol], (NSIMS, 1)) + errVol
    
    # simulation of normalized stock returns
    # simulation of precision and standard errors using Bayesian posterior
    simPrecUSA = np.random.gamma(N/2, 2*stdUSA**(-2)/N, NSIMS)
    simStdUSA = np.power(simPrecUSA, -0.5)
    
    # simulation of regression coefficients using Bayesian posterior
    errUSA = np.random.multivariate_normal([0,0],covUSA,NSIMS) * np.transpose(np.tile(simStdUSA, (2,1)))
    simCoeffUSA = np.tile([slopeUSA, intUSA], (NSIMS, 1)) + errUSA
    
    # output is random regression coefficients
    alpha = simCoeffVol[:, 0]
    beta = simCoeffVol[:, 1]
    theta = simCoeffUSA[:, 0]
    gamma = simCoeffUSA[:, 1]
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = alpha * np.ones(NSIMS) + beta * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetUSA[t] = gamma * np.ones(NSIMS) + theta * simVol[t + 1]  + simVol[t + 1] * noiseUSA[t]
        
    return simRetUSA

T = 30
initVol = 20
model0 = classicSim(initVol, T)
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