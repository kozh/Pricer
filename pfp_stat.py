import numpy as np
import pandas as pd
from datetime import datetime
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def ba_scenarios(BAs, simulation_years, n_scenarios, print_statistics = False, points_in_year = 12):

    days_in_year = 260
    c01  = {0: 1.9, 1: 4.5, 2:1.4, 3:4, 4:1.7}

    BAs = [(x.lower()) for x in BAs]

    n_timepoints = points_in_year*simulation_years
    dt = days_in_year/points_in_year

    Hist = pd.read_excel('inputs/quotes daily.xlsx', 'quotes', decimal = '.')
    Hist.columns = Hist.columns.str.lower()

    Hist = Hist[[(x.lower()) for x in BAs]+['date']]
    Hist.index = Hist.date
    Hist = (Hist[Hist.index.dayofweek <5]).drop('date', 1)

    Returns = np.log((Hist/Hist.shift(1)).astype(np.float32))

    RDiv = pd.read_excel('inputs/quotes daily.xlsx', 'details', decimal = '.')
    RDiv.index = [(x.lower()) for x in RDiv.index]

    mu = (RDiv.loc[BAs])['r+div']/days_in_year
    std = Returns[BAs].std()
    var = Returns[BAs].var()
    cov = Returns[BAs].cov()

    #corr = np.power(Returns[BAs].corr(),1)
    corr = (Returns[BAs].dropna()).corr()

    corr2 = corr
    for i in range(len(BAs)-1):
        for j in range(i+1,len(BAs)-1):
            print(i)
            print(j)
            print((((Returns[[BAs[i+1],BAs[j+1]]]).dropna()).corr()).iloc[0,1])
            corr2.iloc[i+1,j+1] = (((Returns[[BAs[i+1],BAs[j+1]]]).dropna()).corr()).iloc[0,1]

    print(corr2)




    drift = mu - (0.5*var)

    r1 = np.random.multivariate_normal(np.zeros_like(mu), corr, n_scenarios*n_timepoints)
    randoms = r1.reshape(n_timepoints, n_scenarios, len(mu))

    pre_returns = np.zeros_like(randoms)
    for i in range(len(mu)):
        pre_returns[:,:,i] = drift[i]*dt + std[i]*randoms[:,:,i]*(dt**0.5)
    returns = np.exp(pre_returns.cumsum(axis = 0))

    if print_statistics:
        print('Means: \n' + str(mu*252))      
        print(' ')
        print('Sigmas: \n' + str(Returns[BAs].std()*16))
        print(' ')
        print('Correlations: \n')
        print(corr)
        print(' ')
        print('Correlation power: \n')
        print(np.linalg.det(Returns[BAs].corr()))
    return returns

def rfr(currency, term):

    curves = pd.read_excel('inputs/curves.xlsx', 'Sheet1', decimal = '.')
    rfr = curves[term][currency]

    return rfr

def check_stat(BAs):
    if type(BAs) != str:
        return False

    else:

            BAs = [(x.lower()) for x in BAs.split(', ')]

            Hist = pd.read_excel('inputs/quotes daily.xlsx', 'quotes', decimal = '.')
            Hist.columns = Hist.columns.str.lower()

            #row.BAs.split(', ')
            return True


