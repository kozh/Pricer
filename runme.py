from pfp_stat import *

import numpy as np
import pandas as pd 

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib

import datetime as dat

from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF
import fpdf

from pfp_products import *

import os.path


print('...initialiazing products...')
xls = pd.read_excel('inputs/prod_cat full.xlsx', 'prod cat', 
                         decimal = '.', usecols = 'b:z', 
                         index_col = 0, header = 1)

products = xls.transpose()
all_BAs = []
prod_list = []
max_term = 0

for prod, row in products.iterrows():
    
    check = check_stat(row.BAs)
    
    if check:        
        all_BAs.extend(row.BAs.split(', '))
        all_BAs = list(set(all_BAs))
        
        if row.term > max_term: max_term = row.term

        Note1 = Structure(prod, row.BAs, row.notional_curr, row.term, 
                  row.coupon_value, row.coupon_always, row.coupon_check_months, row.coupon_memory, 
                  row.coupon_lower_barrier, row.coupon_upper_barrier,
                  row.autocall_flag, row.autocall_check_months, row.autocall_barrier, row.autocall_barrier_increase_rate,
                  row.redemption_amount, row.redemption_put_strike, row.redemption_guarantee_rule,
                  row.redemption_upside_participation, row.redemption_downside_participation, row.issuer)
        Note1.stats_ok = True
        
        prod_list.append(Note1)
        #print ('We have enough statistics for' + Note1.name + '.')
    else: 
        print ('We don`t have enough statistics for ' + row.name + 
               '`s underlyings.\nWe`ve put it in wishlist and we do not calculate it now.')
print('All other products from `prod cat.xls` are successfully loaded.')


n_scenarios = 50000
simulation_years = max_term   

returns = ba_scenarios(all_BAs, 
                       simulation_years,  
                       n_scenarios, 
                       print_statistics = False)

for prod in prod_list:
    print('calculating ' + prod.name)
    a1 = prod.payoff(all_BAs, returns, True)
    
print('Done!')