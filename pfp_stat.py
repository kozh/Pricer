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

def virr(cfs, precision = 0.0005, rmin = -1, rmax = 1):
    ''' 
    Vectorized IRR calculator. First calculate a 3D array of the discounted
    cash flows along cash flow series, time period, and discount rate. Sum over time to 
    collapse to a 2D array which gives the NPV along a range of discount rates 
    for each cash flow series. Next, find crossover where NPV is zero--corresponds
    to the lowest real IRR value. For performance, negative IRRs are not calculated
    -- returns "-1", and values are only calculated to an acceptable precision.

    IN:
        cfs - numpy 2d array - rows are cash flow series, cols are time periods
        precision - level of accuracy for the inner IRR band eg 0.005%
        rmin - lower bound of the inner IRR band eg 0%
        rmax1 - upper bound of the inner IRR band eg 30%
        rmax2 - upper bound of the outer IRR band. eg 50% Values in the outer 
                band are calculated to 1% precision, IRRs outside the upper band 
                return the rmax2 value
    OUT:
        r - numpy column array of IRRs for cash flow series
    '''

    cfs = np.transpose(cfs)

    if cfs.ndim == 1: 
        cfs = cfs.reshape(1,len(cfs))

    # Range of time periods
    years = np.arange(0,cfs.shape[1])

    # Range of the discount rates

    rates = np.linspace(rmin,rmax,(rmax - rmin)/precision)
    rates[rates==0] = 0.0001

    # Discount rate multiplier rows are years, cols are rates
    drm = (1+rates)**-years[:,np.newaxis]

    # Calculate discounted cfs   
    discounted_cfs = cfs[:,:,np.newaxis] * drm

    # Calculate NPV array by summing over discounted cashflows
    npv = discounted_cfs.sum(axis = 1)

    ## Find where the NPV changes sign, implies an IRR solution
    signs = npv < 0

    # Find the pairwise differences in boolean values when sign crosses over, the
    # pairwise diff will be True
    crossovers = np.diff(signs,1,1)

    # Extract the irr from the first crossover for each row
    irr = np.min(np.ma.masked_equal(rates[1:]* crossovers,0),1)

    # Error handling, negative irrs are returned as "-1", IRRs greater than rmax2 are
    # returned as rmax2
    
    negative_irrs = cfs.sum(1) < 0
    r = np.where(negative_irrs,-1,irr)
    r = np.where(irr.mask * (negative_irrs == False), 0.5, r)

    return r
