import pandas as pd
import numpy as np
from statsmodels.api import OLS
from scipy import stats
from matplotlib import pyplot as plt
from YJX import YJinv

np.random.seed(0)
DF = pd.read_excel('full-data.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]
N = len(vol)
price = DF['Price'].values
dividend = DF['Dividends'].values[1:]
lvol = np.log(vol)
YJresults = stats.yeojohnson(lvol)
la = YJresults[1]
nvol = YJresults[0]
total = np.array([np.log(price[k+1] + dividend[k]) - np.log(price[k]) for k in range(N)])
nUSAret = total/vol
RegVol = OLS(nvol[1:], pd.DataFrame({'const' : 1, 'lag' : nvol[:-1]})).fit()
RegUSA = OLS(nUSAret, pd.DataFrame({'const' : 1/vol, 'vol' : 1})).fit()
intVol = RegVol.params['const']
slopeVol = RegVol.params['lag']
stdVol = np.std(RegVol.resid)
intUSA = RegUSA.params['const']
slopeUSA = RegUSA.params['vol']
stdUSA = np.std(RegUSA.resid)

covVol = np.linalg.inv(np.array([[N - 1, np.sum(nvol[:-1])], [np.sum(nvol[:-1]), np.sum(np.square(nvol[:-1]))]]))
covUSA = np.linalg.inv(np.array([[N, np.sum(1/vol)], [np.sum(1/vol), np.sum(np.square(1/vol))]])) 
print(covVol)
print(covUSA)
NSIMS = 10000

def classicSim(initVol, T):
    ninitVol = stats.yeojohnson(np.log(initVol), la)
    noiseUSA = np.random.normal(0, stdUSA, (T, NSIMS))
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS))
    simRetUSA = np.zeros((T, NSIMS))
    simNVol = np.zeros((T+1, NSIMS))
    simNVol[0] = ninitVol * np.ones(NSIMS)
    
    # now comes the simulation itself!
    # simulate transfomed log volatility as autoregression
    for t in range(T):
        simNVol[t + 1] = intVol * np.ones(NSIMS) + slopeVol * simNVol[t] + noiseVol[t]
    
    # transform back to get volatility
    for t in range(T):
        newVol = np.exp(YJinv(simNVol[t + 1], la))
        simRetUSA[t] = intUSA * np.ones(NSIMS) + slopeUSA * newVol  + newVol * noiseUSA[t]
        
    return simRetUSA

def bayesCoeffSim(initVol, T):
    ninitVol = stats.yeojohnson(np.log(initVol), la)
    noiseUSA = np.random.normal(0, stdUSA, (T, NSIMS))
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS))
    simRetUSA = np.zeros((T, NSIMS))
    simNVol = np.zeros((T+1, NSIMS))
    simNVol[0] = ninitVol * np.ones(NSIMS)
    
    simCoeffVol = np.random.multivariate_normal([intVol, slopeVol], covVol * stdVol**2, NSIMS)
    simCoeffUSA = np.random.multivariate_normal([slopeUSA, intUSA], covUSA * stdUSA**2, NSIMS)
    
    alpha = simCoeffVol[:, 0]
    beta = simCoeffVol[:, 1]
    theta = simCoeffUSA[:, 0]
    gamma = simCoeffUSA[:, 1]
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simNVol[t + 1] = alpha * np.ones(NSIMS) + beta * simNVol[t] + noiseVol[t]
    
    # transform back to get volatility
    for t in range(T):
        newVol = np.exp(YJinv(simNVol[t + 1], la))
        simRetUSA[t] = gamma * np.ones(NSIMS) + theta * newVol  + newVol * noiseUSA[t]
      
    return simRetUSA

def bayesAllSim(initVol, T):
    ninitVol = stats.yeojohnson(np.log(initVol), la)
    noiseUSA = np.random.normal(0, stdUSA, (T, NSIMS))
    noiseVol = np.random.normal(0, stdVol, (T, NSIMS))
    simRetUSA = np.zeros((T, NSIMS))
    simNVol = np.zeros((T+1, NSIMS))
    simNVol[0] = ninitVol * np.ones(NSIMS)
    
    simPrecVol = np.random.gamma((N - 1)/2, 2*stdVol**(-2)/(N - 1), NSIMS)
    simStdVol = np.power(simPrecVol, -0.5)
    simCoeffVol = np.tile([intVol, slopeVol], (NSIMS, 1)) + np.random.multivariate_normal([0, 0], covVol, NSIMS) * np.transpose(np.tile(simStdVol, (2, 1)))
    simPrecUSA = np.random.gamma(N/2, 2*stdUSA**(-2)/N, NSIMS)
    simStdUSA = np.power(simPrecUSA, -0.5)
    simCoeffUSA = np.tile([slopeUSA, intUSA], (NSIMS, 1)) + np.random.multivariate_normal([0, 0], covUSA, NSIMS) * np.transpose(np.tile(simStdUSA, (2, 1)))
    
    alpha = simCoeffVol[:, 0]
    beta = simCoeffVol[:, 1]
    theta = simCoeffUSA[:, 0]
    gamma = simCoeffUSA[:, 1]
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simNVol[t + 1] = alpha * np.ones(NSIMS) + beta * simNVol[t] + noiseVol[t]
    
    # transform back to get volatility
    for t in range(T):
        newVol = np.exp(YJinv(simNVol[t + 1], la))
        simRetUSA[t] = gamma * np.ones(NSIMS) + theta * newVol  + newVol * noiseUSA[t]
              
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
        print('withdrawal rate ', withdrawal)
        print(np.sum(wealth[T] > 0)/NSIMS)