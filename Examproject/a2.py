import numpy as np
from scipy import optimize


"""wage_bar"""
def wage_bar(w, tau):
    return (1-tau) * w



"""consumption"""
def consumption(L, w, tau, kappa, v):
    return kappa - wage_bar(L, w, tau, kappa, v)



"""utility"""
def utility(L, v, G, alpha, C):
    return np.log(C ** alpha * G ** (1-alpha) - v * L ** 2 / 2)



"""value of choice"""
def value_of_choice(L, v, G, w, tau, kappa, alpha):

    C = consumption(L, w, tau, kappa)

    return utility(L, v, G, alpha, C)


"""optimal labor supply"""
def opti_labor(v, G, w, tau, kappa, alpha):
        

    obj = lambda L: value_of_choice(L, v, G, w, tau, kappa, alpha)
    res = optimize.minimize_scalar(obj,bounds=(0,1),method='bounded')

    return res.x