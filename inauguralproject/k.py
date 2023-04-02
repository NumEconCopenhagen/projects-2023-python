

    def calc_utility_addition(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # (i) defining the aggregated consumption
        C = par.wM * LM + par.wF * LF

        # (ii) setting an if-function
        if par.sigma == 1:
           H = HM ** (1 - par.alpha) * HF ** par.alpha
        elif par.sigma == 0:
           H = np.minimum(HM, HF)
        else:
          H = ((1 - par.alpha) * HM ** ((par.sigma - 1) / par.sigma) + par.alpha * HF ** ((par.sigma - 1) / par.sigma)) ** (par.sigma / (par.sigma - 1))

        # (iii) defining aggregated utility
        Q = C ** par.omega * H ** (1 - par.omega)
        utility = np.fmax(Q, 1e-8) ** (1 - par.rho) / (1 - par.rho) - 0.2 * LM + 0.2 * LF + 0.4 * HM + 0.4 * LF

        # (iv) defining disutility of working
        epsilon_ = 1 + 1 / par.epsilon_
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu * (TM ** epsilon_ / epsilon_ + TF ** epsilon_ / epsilon_)
        
        return utility - disutility
    





    def solve_wF_vec_addition(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol
        max = SimpleNamespace()

        # (i) Looping trough wF_vec
        for i, wF in enumerate(par.wF_vec):
           
           # (ii) defining the values of work of females
           par.wF = wF

           # (iii) defining the objective function
           obj = lambda x: -self.calc_utility_addition(x[0], x[1], x[2],x[3])
        
           # (iv) setting the initial guess
           guess = [1, 1, 1, 1]

           # (v) defining the constraints
           constraints = ({'type': 'ineq', 'fun': lambda x :  -x[0] - x[1] + 24}, {'type': 'ineq', 'fun': lambda x : -x[2] -x[3] + 24})

           # (vi) defining the bounds
           bounds = ((1e-8, 24), (1e-8, 24), (1e-8, 24), (1e-8, 24))

           # (vii) defining the solution by the minimizing function
           res = optimize.minimize(obj, guess, method='SLSQP', bounds=bounds, constraints=constraints)

           # (iix) defining and stacking HM and HF
           max.HM = res.x[1]
           max.HF = res.x[3]
           sol.HM_vec[i] = res.x[1]
           sol.HF_vec[i] = res.x[3]
        

        return sol.HM_vec, sol.HF_vec







    def objective_addition(self,x):
        """ calculate the objective function"""

        par = self.par
        sol = self.sol
        
        # (i) defining sigma
        par.sigma = x
        
        # (ii) defining optimal home labour
        solver = self.solve_wF_vec_addition()
        
        # (iii) defining the regression
        regression = self.run_regression()
        
        # (iv) defining the objective function
        obj = (par.beta_0_target - sol.beta_0) ** 2 + (par.beta_1_target - sol.beta_1) ** 2

        return obj
    


    def estimate_addition(self,alpha=None,sigma=None):
        """ sigma """

        par = self.par 
        sol = self.sol
        alpha_sigma=SimpleNamespace()
        
        # (i) defining the guess
        guess = 1

        # (ii) setting the bounds
        bounds = ((0,5))
        
        # (iii) applying the minimization function
        res = optimize.minimize_scalar(self.objective_addition, guess, bounds=bounds, method='bounded')
        
        # (iv) saving and printing the results 
        alpha_sigma.sigma = res.x
        print(f'optimal sigma = {alpha_sigma.sigma:.4f}')

        return 