from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta_0_target = 0.4
        par.beta_1_target = -0.1

        # e.1 addition to the model
        

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta_0 = np.nan
        sol.beta_1 = np.nan



    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM * LM + par.wF * LF

        # b. home production
        if par.sigma == 1:
           H = HM ** (1-par.alpha) * HF ** par.alpha
           
        elif par.sigma == 0:
           H = np.minimum(HM, HF)
           
        else:
          H = ((1-par.alpha)*HM**((par.sigma -1 )/par.sigma) + par.alpha*HF**((par.sigma -1 )/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C ** par.omega * H ** (1-par.omega)
        utility = np.fmax(Q,1e-8) ** (1-par.rho) / (1-par.rho)

        # d. disutlity of work
        epsilon_ = 1 + 1 / par.epsilon
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu * (TM ** epsilon_ / epsilon_ + TF ** epsilon_ / epsilon_)
        
        return utility - disutility

 
    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        

        # a. all possible choices
        x = np.linspace(0, 24, 49)
        LM, HM, LF, HF = np.meshgrid(x, x, x, x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM, HM, LF, HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM + HM > 24) | (LF + HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k, v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt




    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # (i) Defining objective function
        def objective(x):
            return -self.calc_utility((x[0]), x[1], x[2], x[3])
        obj = lambda x: objective(x)
        
        # (ii) Defining the guess
        guess = [4, 4.2, 5, 5.2]

        # (iii) Defining the contraints 
        constraints =   ({'type': 'ineq', 'fun': lambda x :  -x[0] - x[1] + 24}, {'type': 'ineq', 'fun': lambda x : -x[2] -x[3] + 24})

        # (iv) Set the bounds
        bounds = ((1e-8,24), (1e-8,24), (1e-8,24), (1e-8,24))

        # (v) Applying the minimize function
        res = optimize.minimize(obj, guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # (vi) Saving the results
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        return opt 





    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par=self.par
        sol=self.sol
        max=SimpleNamespace()

        # (a) Looping
        for i, wF in enumerate(par.wF_vec):
           
           # (i) defining the parameters
           par.wF = wF

           # (ii) defining the objective function
           obj = lambda x: -self.calc_utility(x[0], x[1], x[2],x[3])
        
           # (iii) Setting the initial guess
           guess = [4, 4.2, 5, 5.2]

           # (iv) Setting the constraints
           constraints = ({'type': 'ineq', 'fun': lambda x :  -x[0] - x[1] + 24}, {'type': 'ineq', 'fun': lambda x : -x[2] -x[3] + 24})

           # (v) Setting the bounds
           bounds = ((1e-8,24), (1e-8,24), (1e-8,24), (1e-8,24))

           # (vi) defining the solutiong by the minimizing function
           res = optimize.minimize(obj, guess, method='SLSQP', bounds=bounds, constraints=constraints)

           # (vii) finding and stacking the results
           max.HM = res.x[1]
           max.HF = res.x[3]
           sol.HM_vec[i] = res.x[1]
           sol.HF_vec[i] = res.x[3]
        
        return sol.HM_vec, sol.HF_vec



    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec / sol.HM_vec)
        A = np.vstack([np.ones(x.size), x]).T
        sol.beta_0, sol.beta_1 = np.linalg.lstsq(A, y, rcond = None)[0]



    def objective(self,x):
        """ calculate the objective function"""

        par = self.par
        sol = self.sol
        
        # (i) Updating alpha and sigma
        par.alpha = x[0]
        par.sigma = x[1]
        
        # (ii) Solving optimal hours worked in home
        solver = self.solve_wF_vec()
        
        # (iii) Estimating the regression
        regression = self.run_regression()
        
        # (iv) Calculating the function
        obj = (par.beta_0_target - sol.beta_0)**2 + (par.beta_1_target - sol.beta_1)**2

        return obj
    


    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol
        alpha_sigma=SimpleNamespace()
        
        # (i) Setting the guess
        guess = [0.5, 0.5]

        # (ii) Defining the bounds for alpha and sigma
        bounds = ((0, 1), (0, None))
        
        # (iii) defining the solution by the minimizing function
        res = optimize.minimize(self.objective, guess, bounds=bounds, method='SLSQP')
        
        # (iv) Saving and printing the reults
        alpha_sigma.alpha = res.x[0]
        alpha_sigma.sigma = res.x[1]
        print(f'optimal alpha = {alpha_sigma.alpha:.4f}')
        print(f'optimal sigma = {alpha_sigma.sigma:.4f}')

        return res
    



