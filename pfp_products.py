import numpy as np
import pandas as pd
import plotly 
import plotly.plotly as py
import matplotlib
import matplotlib.mlab as mlab
import random
import time
import datetime
import os
import matplotlib.pyplot as plt

from pfp_stat import rfr
from timeit import default_timer as timer

from datetime import datetime
from scipy import stats
from scipy.linalg import svd, sqrtm, cholesky
from scipy.stats import multivariate_normal, normaltest, shapiro

from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig

from fpdf import FPDF
import fpdf

points_in_year = 12

class Structure:
	def __init__(self, name, BAs, notional_curr, term, 
		coupon_value, coupon_always, coupon_check_months, coupon_memory, coupon_lower_barrier, coupon_upper_barrier,
		autocall_flag, autocall_check_months, autocall_barrier, autocall_barrier_increase_rate,
		redemption_amount, redemption_put_strike, redemption_guarantee_rule, redemption_upside_participation, redemption_downside_participation, 
		issuer):

		self.issuer = issuer
		self.name = name
		self.BAs = BAs.split(', ') 		# базовые активы
		self.notional_curr = notional_curr # валюта номинала ноты
		self.term = term 	# срок жизни ноты в годах

		self.coupon_value = coupon_value
		self.coupon_always = coupon_always
		self.coupon_check_months = coupon_check_months	# частота проверки купона и условия автокола
		self.coupon_memory = coupon_memory
		self.coupon_lower_barrier = coupon_lower_barrier
		self.coupon_upper_barrier = coupon_upper_barrier

		self.autocall_flag = autocall_flag
		self.autocall_check_months = autocall_check_months	# частота проверки купона и условия автокола
		self.autocall_barrier = autocall_barrier
		self.autocall_barrier_increase_rate = autocall_barrier_increase_rate

		self.redemption_put_strike = redemption_put_strike
		self.redemption_amount = redemption_amount
		self.redemption_guarantee_rule = redemption_guarantee_rule
		self.redemption_upside_participation = redemption_upside_participation
		self.redemption_downside_participation = redemption_downside_participation


		self.check_months = self.autocall_check_months


		self.rfr = rfr(self.notional_curr, term)		# безрисковая ставка в валюте номинала ноты

		self.n_points = int(self.term*points_in_year/self.check_months)   # сколько точек проверки за срок жизни инструмента
		self.time_steps = np.arange(self.check_months - 1, self.term*points_in_year, self.check_months) #прожитых месяцев на каждом шаге

		self.stat_ok = 0

		# 1: worst < put_strike => worst; worst > put_strike => redemption_amount
		# 2: min(1, max(0, worst/strike))
	
	def discounter(self, n_scenarios):

		discount_base = np.exp(-self.rfr*(self.time_steps + 1)/12)
		discount_base = (1+self.rfr)**(-(self.time_steps + 1)/12)
		discount = np.reshape(np.repeat(discount_base, n_scenarios), (self.n_points, n_scenarios)) 

		return discount

	def make_pdf(self, a1):

		pdf = FPDF()
		pdf.add_page()
		pdf.set_xy(10, 15)
		pdf.set_font('arial', 'B', 16)
		pdf.cell(0, 10, 'Instrument description', 0, 2, 'C')

		pdf.set_font('arial', '', 12)
		pdf.set_x(15)
		pdf.cell(0, 6, 'Issuer: ' + self.issuer, 0, 2, 'L')
		pdf.cell(0, 6, 'ISIN: ' + self.name, 0, 2, 'L')
		pdf.cell(0, 6, '', 0, 2, 'L')

		pdf.cell(0, 6, 'Currency: ' + self.notional_curr, 0, 2, 'L')
		pdf.cell(0, 6, 'Term: ' + str(self.term) + (' year ' if self.term == 1 else ' years'), 0, 2, 'L')
		pdf.cell(0, 6, ('Asset: ' if len(self.BAs) == 1 else 'Assets: ') + ', '.join(self.BAs), 0, 2, 'L')
		pdf.cell(0, 6, '', 0, 2, 'L')

		if self.coupon_value == 0:
			pdf.cell(0, 6, 'Coupon: no\n', 0, 2, 'L')

		if self.coupon_value >0:
			txt_coupon = (('Coupon: %.2f%%' % (self.coupon_value*100))
				+ ((' every %d months\n' % self.check_months) if self.check_months > 1 else ' every month\n'))
				
			if self.coupon_always == 1:
				pdf.cell(0, 6, txt_coupon, 0, 2, 'L')
			if self.coupon_always == 0:
				txt2 = ('Conditional coupon: yes\nMemory effect: ' 
					+ ('yes' if self.coupon_memory == 1 else 'no') 
					+ '\nCoupon barrier: %.f%%' % (self.coupon_lower_barrier*100))
				pdf.multi_cell(75, 6, txt_coupon + txt2, 1, 2, 'L')

			pdf.cell(0, 6, '', 0, 2, 'L')  
			pdf.set_xy(15, pdf.get_y())

			if self.autocall_flag == 0:
				pdf.cell(0, 6, 'Autocallable: no\n', 0, 2, 'L')
			else:
				txt2 = (('Autocallable: yes\n') + 
					(('Autocall level: %.f%%\n' % (self.autocall_barrier*100)) if self.autocall_barrier_increase_rate == 0 
						else
							(('Initial autocall level: %.f%%\n' % (self.autocall_barrier*100)) +

							(('Autocall level increase step: %.f%%\n' % (self.autocall_barrier_increase_rate*100)) 
								if self.autocall_barrier_increase_rate >0 else
								('Autocall level decrease step: %.f%%\n' % (-self.autocall_barrier_increase_rate*100)) 
								))))
		        
				pdf.multi_cell(75, 6, txt2, 1, 2, 'L')

		# погашение:

		txt0 = ('Capital protection level: %.f%%\n' % (self.redemption_amount*100)) 

		if self.redemption_put_strike > 0:
			txt1 = (('Capital protection barrier: %.f%%\n' % (self.redemption_put_strike*100)) +
					('Redemption rule below barrier: ') +
					('worst performance' if self.redemption_guarantee_rule == 1 
						else (
							'(worst performance) / (protection barrier)' if self.redemption_guarantee_rule == 2 
								else (
								''
							 ))))
		else: 
			txt1 = ''

		txt21 = (('Downside participation: %.f%%\n' % (self.redemption_downside_participation*100)) 
			if self.redemption_downside_participation >0 else '')
		txt22 = (('Upside participation: %.f%%\n' % (self.redemption_upside_participation*100)) 
			if self.redemption_upside_participation >0 else '')

		txt = txt0 + txt1 + txt21 + txt22

		pdf.cell(0, 6, '', 0, 2, 'L')  
		pdf.set_xy(15, pdf.get_y())
		pdf.multi_cell((75+75*(txt1!='')), 6, txt, (txt != txt0), 2, 'L')

		# погашение - закончили

		pdf.cell(0, 6, '', 0, 2, 'L')  
		pdf.set_xy(15, pdf.get_y())

		pdf.cell(0, 6, ('Theoretical price: %.2f%%' % (a1.sum(axis = 0).mean()*100)), 0, 2, 'L')
		#pdf.cell(0, 6, ('Max payoff: %.2f%%' % ((a1.sum(axis = 0).max())*100)), 0, 2, 'L')

		pdf.cell(0, 6, '', 0, 2, 'L')    

		pdf.cell(90, 10, " ", 0, 2, 'C')
		pdf.cell(-30)

		fig = figure(figsize=(16, 10))
		plt.yticks([])
		plt.xticks(fontname="Arial", fontsize=24)

		chart_name = self.name + ' NPV distribution'
		title(self.name + ' NPV distribution', fontname="Arial", fontsize=24)
		bins = np.linspace(0,1.5,75)
		plt.hist(a1.sum(axis = 0), bins, alpha=1, color = 'salmon')
		plt.savefig('results/' + self.name + ' payoff.png')
		plt.close(fig)
		pdf.image('results/' + self.name + ' payoff.png', x = 2, y = 170, w = 200, h = 0, type = '', link = '')
		pdf.output('results/' + self.name + ' description sheet', 'F')
		os.remove('results/' + self.name + ' payoff.png')


	def payoff(self, underlyings, returns, to_pdf = False):

		self.underlyings = underlyings
		self.returns = returns

		positions = []
		for a in self.BAs:
			positions.append(self.underlyings.index(a))

		n_scenarios = self.returns.shape[1]

		rtrns = (self.returns[:,:,positions])[self.time_steps,:,:]

		############################################################################################
		# автоколл
		############################################################################################
		#
		active_flag = np.ones((rtrns.shape[0],rtrns.shape[1]))
		autocalled = np.zeros((rtrns.shape[0],rtrns.shape[1]))
		
		# active_flag.shape - основной размер массива для дальнейших операций

		if self.autocall_flag == 1:
			c1 = self.autocall_barrier + self.autocall_barrier_increase_rate*np.array(range(0,self.n_points))
			call_trigger = np.reshape(np.repeat(c1, n_scenarios*len(self.BAs)), (self.n_points, n_scenarios, len(self.BAs)))  
			call_flag = (rtrns > call_trigger).all(axis = 2)
			c0 = call_flag
			c1 = c0.cumsum(axis = 0)
			c2 = (c1 > 0)*1 
			c3 = np.roll(c2, 1, axis = 0)
			c3[0,:] = 0
			call_flag[-1,:] = 0
			active_flag = 1 - c3
			autocalled = active_flag*call_flag
		#
		############################################################################################
		# автоколл - сделано
		############################################################################################

		############################################################################################
		# купоны
		############################################################################################
		cpns = np.zeros(active_flag.shape)
		if self.coupon_value >0:
			if self.coupon_always == 1:
				cpns = np.ones(active_flag.shape)

			else:
				cpns = ((rtrns.min(axis = 2) > self.coupon_lower_barrier) * 
						(rtrns.min(axis = 2) < self.coupon_upper_barrier))

				if self.coupon_memory == 1:        
					a3 = pd.DataFrame(np.zeros(cpns.shape))
					for i in range(0,self.n_points):
						a3.ix[i] = ((i+1) - a3.ix[:i].sum(axis = 0))*pd.DataFrame(cpns).ix[i]
					cpns = a3

		coupons = cpns*active_flag*self.coupon_value

		############################################################################################
		# купоны - сделано
		############################################################################################

		############################################################################################
		# на последний день
		############################################################################################

		# типы погашений:
		# 1: worst < put_strike => worst; worst > put_strike => redemption_amount
		# 2: min(1, worst/strike)

		guarantee_payoff = np.zeros(active_flag.shape)
		guarantee_payoff[-1,:] = self.redemption_amount

		wrst = rtrns.min(axis = 2)
		worst_last = wrst[-1,:]

		if self.redemption_guarantee_rule == 1:			
			guarantee_payoff[-1,:] = ((worst_last < self.redemption_put_strike)*worst_last +
				(worst_last >= self.redemption_put_strike)*(worst_last < self.redemption_amount) + 
				(worst_last >= self.redemption_amount)*(self.redemption_amount + 
					np.abs(worst_last - self.redemption_amount)*self.redemption_upside_participation))

		if self.redemption_guarantee_rule == 2:
			guarantee_payoff[-1,:] = ((worst_last < self.redemption_put_strike)*worst_last/self.redemption_put_strike + 
				(worst_last >= self.redemption_put_strike)*(worst_last < self.redemption_amount) + 
				(worst_last >= self.redemption_amount)*(self.redemption_amount + 
					np.abs(worst_last - self.redemption_amount)*self.redemption_upside_participation))

		guarantee_payoff = guarantee_payoff*active_flag

		############################################################################################
		# на последний день - сделано
		############################################################################################

		total_payoff = coupons + autocalled + guarantee_payoff	
		total_discounted_payoff = total_payoff * self.discounter(n_scenarios)

		if to_pdf == True:
			self.make_pdf(total_discounted_payoff)

		return total_discounted_payoff


