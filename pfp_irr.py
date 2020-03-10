import numpy as np
import pandas as pd
from datetime import datetime

def cf_portf(Cf, w):
	return  Cf*w

def vector_irr(cf):
    a1 = (payoffs*wghts).sum(axis = 2)    
    a2 = np.ones((a1.shape[0]+1,a1.shape[1]))*(-1)
    a2[1:,:] = a1
    a3 = np.array([np.irr(x) for x in np.transpose(a2)]) 
    return a3

