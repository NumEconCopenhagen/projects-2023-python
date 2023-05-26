# (i) Importing packages
import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
from types import SimpleNamespace




""" Defining the non-dynamic model (quesiton 1)"""
class hairdresser:
        
    def __init__(self, **kwargs):

        # (i) create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # (ii) defining parameters
        par.eta = 0.5                                  # elasticity of demand
        par.w = 1.0                                    # wage
        par.kappa_t = np.linspace(1.0, 2.0)            # demand shock which can take the values from 1-2
    
        # (iii) solution
        sol.l_t = np.nan



    """ Negative profit """
    def negativeprofit(self, l_t, kappa_t):
        par = self.par
        sol = self.par

        # (ii) production
        y_t = l_t

        # (iii) price per haircut
        p_t = kappa_t * y_t ** (- par.eta)

        # (iv) profit
        pi_t = p_t * y_t - par.w * l_t 

        return -pi_t



    """ Defining the optimal labor """
    def optimal_l_t(self):

        par =self.par

        # (i) Defining an empty list
        optimal_l_t_values = []

        # (ii) Defining a loop for l_t over all kappa values
        for kappa_t_value in par.kappa_t:
            l_t = ((1 - par.eta) * kappa_t_value / par.w) ** (1 / par.eta)
            optimal_l_t_values.append(l_t)

        # (iii) return
        return optimal_l_t_values
    


    """ Defining the maximization problem """
    def opti_employees(self):

        par = self.par
        sol = self.sol

        # (i) Defining the solution as a dictionary over all kappa values
        sols = dict()
        for kappa_t in par.kappa_t:

            # (i.a) Guess for l_t
            l_guess = 0

            # (i.b) Creating a pa
            tmp_util = lambda l_t: self.negativeprofit(l_t,kappa_t=kappa_t)
            sols[str(kappa_t)] = optimize.minimize(tmp_util, l_guess, bounds =((0,None),), method='Nelder-Mead')#['x'][0]



        # (ii) return
        return sols
    






""" Defining the dynamic model (quesiton 2-5)"""
class dynamic_hairdresser:

    """ Setting the initial parameters """
    def __init__(self, **kwargs):

        # (i) create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # (ii) defining parameters
        par.eta = 0.5                                  # elasticity of demand
        par.w = 1.0                                    # wage
        #par.kappa_t = np.linspace(1.0, 2.0)            # demand shock
        par.rho = 0.9
        par.iota = 0.01
        par.sigma_e = 0.1
        par.R = (1+0.01)**(1/12)

        # (iii) Determine the time period to be 120 months
        self.T = 120



    """ Defining kappa """
    def kappa_t(self):

        par = self.par

        # (i) defining an array for kappa
        kappas = np.array([])

        # (ii) defining the inital values of k and epsilon
        k_t = 1
        eps = 0

        # (iii) looping
        for t in range(self.T):

            # (i.a) defining k_t
            k_t = np.exp(par.rho * np.log(k_t) + eps)

            # (i.b) epsilon is a random normal distribution
            eps = np.random.normal(loc = -0.5*par.sigma_e**2, scale = par.sigma_e)

            # (i.c) kappa
            kappas = np.append(kappas, k_t) 

        # (iv) return
        return kappas



    """ Defining l_t as a function of kappa_t """
    def l_t(self, kappa_t):

        par = self.par
        
        # return
        return (((1-par.eta)*kappa_t)/par.w)**(1/par.eta)



    """ Defining h1 """
    def h1(self):

        par = self.par
        
        # (i) defining kappas
        kappas = self.kappa_t()

        # (ii) setting the initial values
        sum_ = 0
        l_t = 0

        # (iii) loop
        for i in range(self.T):
            
            # (iii.a) Defining l_t
            new_l_t = self.l_t(kappas[i]) 

            # (iii.b) if-statement
            if new_l_t != l_t:
                sum_ = sum_ + par.R ** (-i) * (kappas[i]*new_l_t**(1-par.eta) - par.w*new_l_t - par.iota)
                l_t = new_l_t
            else:
                sum_ = sum_ + par.R ** (-i) * (kappas[i]*new_l_t**(1-par.eta) - par.w*new_l_t)
                l_t = new_l_t
    
        # (iv) return
        return sum_



    """ Defining H1 as a function of K """
    def H1(self, K):
        par = self.par

        # (i) defining an empty list
        m = []

        # (ii) looping
        for i in range(K):
            m.append(self.h1())

        # (iii) return
        return 1 / K * np.sum(m)
    


    """ Defining h2 as a function of delta """
    def h2(self, delta):

        par = self.par
        
        # (i) defining kappa
        kappas = self.kappa_t()

        # (ii) defining the initial values
        sum_ = 0
        l_t = 0

        # (iii) looping
        for i in range(self.T):
                       
            # (iii.a) defining l_t
            new_l_t = self.l_t(kappas[i]) 

            # (iii.b) if-statement
            if abs(l_t - new_l_t) > delta:
                sum_ = sum_ + par.R ** (-i) * (kappas[i]*new_l_t**(1-par.eta) - par.w*new_l_t - par.iota)
                l_t = new_l_t
            else:
                new_l_t=l_t
                sum_ = sum_ + par.R ** (-i) * (kappas[i]*new_l_t**(1-par.eta) - par.w*new_l_t)
                l_t = new_l_t
    
        # (iv) return
        return sum_
    



    """ Defining H2 as a function of K and delta"""
    def H2(self, K, delta):
        par = self.par

        # (i) empty list
        m = []

        # (ii) looping
        for i in range(K):
            m.append(self.h2(delta))

        # return
        return 1 / K * np.sum(m)
    
      



    """ Defining h3 as a function of delta """
    def h3(self, delta):

        par = self.par
        
        # (i) defining kappa
        kappas = self.kappa_t()

        # (ii) defining the initial values
        sum_ = 0
        l_t = 5

        # (iii) looping
        for i in range(self.T):
                       
            # (iii.a) defining l_t
            new_l_t = self.l_t(kappas[i]) 

            # (iii.b) if-statement
            if abs(l_t - new_l_t) > delta:
                sum_ = sum_ + par.R ** (-i) * (kappas[i]*new_l_t**(1-par.eta) - par.w*new_l_t - par.iota)
                l_t = new_l_t
            else:
                new_l_t=l_t
                sum_ = sum_ + par.R ** (-i) * (kappas[i]*new_l_t**(1-par.eta) - par.w*new_l_t)
                l_t = new_l_t
    
        # (iv) return
        return sum_
    



    """ Defining H3 as a function of K and delta"""
    def H3(self, K, delta):
        par = self.par

        # (i) empty list
        m = []

        # (ii) looping
        for i in range(K):
            m.append(self.h3(delta))

        # return
        return 1 / K * np.sum(m)
    
      