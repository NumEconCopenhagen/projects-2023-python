import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class OLGmodelnummerical():

    def __init__(self, **kwargs):
        
        self.setup_initial()
        self.setup_updated(kwargs)
        self.functionOLG()



    """ Defining the parameters and variables """
    def setup_initial(self):

        # (i) Model
        self.alpha = 0.33
        self.tau = 0.2
        self.A = 20
        self.rho = 0.2
        self.n = 0.01


        # (ii) Transition
        self.kt_min = 1e-10
        self.kt_max = 20
        self.numberdots = 1000
        


    """ Opdating the parameters and variables """
    def setup_updated(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)



    """ Defining the function """
    def functionOLG(self):

        # (i) minimum value of epsilon
        epsilon = 1e-10


        # (ii) steay state
        self.ss = 0


        # (iii) utility function
        self.U_t = lambda c_t: np.log(np.fmax(c_t, epsilon))


        # (iv) production function
        self.Y = lambda k_t: self.A * np.fmax(k_t, epsilon) ** self.alpha
        self.Y_prime = (lambda k_t: self.A * self.alpha * np.fmax(k_t, epsilon) ** (self.alpha - 1))




    """ Defining the optimization problem of the firms"""
    def opti_firms(self, k_t):

        # (i) Gross interest rate
        R_t = self.Y_prime(k_t)


        # (ii) Real rate of wage
        w_t = self.Y(k_t) - self.Y_prime(k_t) * k_t


        # (iii) return
        return R_t, w_t



    """ Defining the consumer's lifetime utility """
    def utility_lifetime(self, c_t, w_t, R_t1, w_t1):
    
        # (i) Consumption (old)
        c_t1 = R_t1 * ((1 - self.tau) * w_t - c_t) + (1 + self.n) * (self.tau * w_t1)


        # (ii) Lifetime tility
        utility = self.U_t(c_t) + (1 / (1 + self.rho)) * self.U_t(c_t1)


        # (iii) return
        return utility



    """ Solving households maximization problem """
    def household_maxproblem(self, w_t, R_t1, w_t1):

        # (i) Minimize funciton
        obj = lambda c_t: -self.utility_lifetime(c_t, w_t, R_t1, w_t1)


        # (ii) Defining optimal consumption
        c_max = (1 - self.tau) * w_t 
        c_t = optimize.fminbound(obj, 0, c_max)


        # (iii) Savings
        s_t = w_t * (1 - self.tau) - c_t


        # (iv) return
        return s_t




    """ Solving households maximization problem """
    def equilibrium(self, k_t1, disp=0):

        # (i) Factor prices in period t+1
        R_t1, w_t1 = self.opti_firms(k_t1)


        # (ii) Minimize function
        def obj(k_t):

            # (ii.a) Factor prices in period t
            _Rt1, w_t = self.opti_firms(k_t)

            # (ii.b) Optimal saving
            s_t = self.household_maxproblem(w_t, R_t1, w_t1)

            # (ii.c) Deviation
            devi = (k_t1 - s_t / (1 + self.n)) ** 2 

            # (ii.d) Return
            return devi
        

        # (iii) Optimal k_t
        kt_max = self.kt_max
        kt_min = 0
        k_t = optimize.fminbound(obj, kt_min, kt_max, disp=disp)


        # (iv) return
        return k_t



    """ Solving transition curve """
    def transitioncurve(self):

        # (i) Capital at t+1
        self.plot_k_t1 = np.linspace(self.kt_min, self.kt_max, self.numberdots)


        # (ii) Capital at t
        self.plot_k_t = np.empty(self.numberdots)
        for i, k_t1 in enumerate(self.plot_k_t1):
            k_t = self.equilibrium(k_t1)
            self.plot_k_t1[i] = k_t
            if (np.abs(k_t1 - k_t) < 0.01 and k_t > 0.01 and k_t < 19):
                self.ss = k_t 



    """ Plotting the transition curve """
    def plot_transition_curve(self, ax, **kwargs):

        ax.plot(self.plot_k_t, self.plot_k_t1, **kwargs)

        ax.set_xlim(0, self.kt_max)
        ax.set_ylim(0, self.kt_max)
        ax.set_xlabel("$k_t$")
        ax.set_ylabel("$k_{t+1}$")



    """ Plotting the 45-degree curve """
    def fourtyfive_curve(self, ax, **kwargs):

        if not "color" in kwargs:
            kwargs["color"] = "black"
        if not "ls" in kwargs:
            kwargs["ls"] = "--"

        ax.plot([self.kt_min, self.kt_max], [self.kt_min, self.kt_max], **kwargs)

