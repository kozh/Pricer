import numpy as np
import pandas as pd
from datetime import datetime
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

def get_hist(BAs, points_in_year, stat_depth):

    BAs = [(x.lower()) for x in BAs]
    Hist = pd.read_excel('inputs/quotes daily.xlsx', 'quotes', decimal = '.')
    Hist.columns = Hist.columns.str.lower()
    Hist = Hist[[(x.lower()) for x in BAs]+['date']]
    Hist.index = Hist.date
    Hist.drop('date', 1, inplace = True)
    Hist.fillna( method ='ffill', inplace = True)
    Hist = (Hist[Hist.index.dayofweek < 5])
    Hist = Hist[Hist.index >=  (Hist.index[-1] - relativedelta(years=stat_depth))]
    bus_days_in_the_period = np.int(np.round(Hist.shape[0]/(stat_depth*points_in_year)))
    Hist = Hist.iloc[::bus_days_in_the_period,:]
    return Hist

def calc_correlations(BAs, points_in_year, stat_depth):

    BAs = [(x.lower()) for x in BAs]

    Hist = pd.read_excel('inputs/quotes daily.xlsx', 'quotes', decimal = '.')
    Hist.columns = Hist.columns.str.lower()
    Hist = Hist[[(x.lower()) for x in BAs]+['date']]
    Hist.index = Hist.date
    Hist.drop('date', 1, inplace = True)
    Hist.fillna( method ='ffill', inplace = True)
    Hist = (Hist[Hist.index.dayofweek < 5])

    # это полная история. теперь берем срезы:

    cors = np.zeros((len(points_in_year), len(stat_depth)))

    for i, points in enumerate(points_in_year):
        for j, depth in enumerate(stat_depth):
            Hist1 = Hist[Hist.index >=  (Hist.index[-1] - relativedelta(years=stat_depth[j]))]
            #Hist1 = Hist[Hist.index >=  (Hist.index[-1] - relativedelta(years=stat_depth[j]))]
            bus_days_in_the_period = np.int(np.round(Hist1.shape[0]/(stat_depth[j]*points_in_year[i])))
            Hist2 = Hist1.iloc[::bus_days_in_the_period,:]
            Returns = np.log((Hist2/Hist2.shift(1)).astype(np.float32))
            corr = Returns[BAs].corr()
            cors[i, j] = corr.iloc[0,1] 
    return cors

def ba_scenarios(BAs, simulation_years, n_scenarios, print_statistics = False, points_in_year = 12, stat_depth = 3):

    BAs = [(x.lower()) for x in BAs]

    n_timepoints = points_in_year*simulation_years

    Hist = get_hist(BAs, points_in_year, stat_depth)
    Returns = np.log((Hist/Hist.shift(1)).astype(np.float32))

    RDiv = pd.read_excel('inputs/quotes daily.xlsx', 'details', decimal = '.')
    RDiv.index = [(x.lower()) for x in RDiv.index]
    
    mu = (RDiv.loc[BAs])['r']/points_in_year

    std = Returns[BAs].std()
    var = Returns[BAs].var()
    corr = Returns[BAs].corr()

    drift = mu - (0.5*var)

    r1 = np.random.multivariate_normal(np.zeros_like(mu), corr, n_scenarios*n_timepoints)
    randoms = r1.reshape(n_timepoints, n_scenarios, len(mu))

    pre_returns = np.zeros_like(randoms)
    for i in range(len(mu)):
        pre_returns[:,:,i] = drift[i] + std[i]*randoms[:,:,i]
    returns = np.exp(pre_returns.cumsum(axis = 0))

    if print_statistics:
        print('Returns: \n' + str(mu*points_in_year))      
        print(' ')
        print('Sigmas: \n' + str(Returns[BAs].std()*(points_in_year**0.5)))
        print(' ')
        print('Correlations: \n')
        print(corr)
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


