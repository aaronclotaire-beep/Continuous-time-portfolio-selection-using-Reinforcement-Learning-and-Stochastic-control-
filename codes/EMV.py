import numpy as np
import pandas as pd
from numpy.random import default_rng
rng = default_rng()



#################################   Stationnary Market simulator   ###############################

def simule_market_statio(T, mu, sigma, M, n, shift_b):
    """
        Simulates the stationary market as a geometric brownian motion.
        Returns:
            tuple: (past_prices, time_vector, future_prices), where:
                - past_prices: samples of the last 100 prices,
                - time_vector: corresponding time steps,
                - future_prices: prices over the investment horizon.
    """
    dt = 1 / n 
    T1 = shift_b*dt
    t1 = np.arange(-T1, 0, dt) 
    t2 = np.arange(0, T+dt, dt) 
    t = np.concatenate((t1,t2))
    W = np.cumsum(rng.standard_normal(size=(M, len(t)-1)) * np.sqrt(dt), axis=1)
    W = np.hstack((np.zeros((M, 1)), W))
    W = W - W[:,shift_b][:,None]
    S = np.exp((mu - 0.5 * sigma**2) * t[None,:] + sigma * W)
    return (S[:,:shift_b], t[shift_b:], S[:,shift_b:])
   
   
###############################   Non Stationnary Market simulator   #############################

def simule_market_non_statio(T, mu0, sigma0, r, gam, delta, M, n):
    """
        Simulates the non-stationary market as described in the article.
        Returns:
            tuple: (past_prices, time_vector, future_prices), where:
                - past_prices: samples of the last 100 prices,
                - time_vector: corresponding time steps,
                - future_prices: prices over the investment horizon.
    """
    dt = 1 / n 
    t = np.arange(0, T+dt, dt) 
    W = np.cumsum(rng.standard_normal(size=(M, len(t)-1)) * np.sqrt(dt), axis=1)
    W = np.hstack((np.zeros((M, 1)), W))
    
    W2 = np.cumsum(rng.standard_normal(size=(M, len(t)-1)) * np.sqrt(dt), axis=1)
    W2 = np.hstack((np.zeros((M, 1)), W2))
    
    W1 = gam*W + np.sqrt(1-gam**2)*W2
    
    sigma = sigma0*np.exp((delta/2)*t[None,:] + np.sqrt(delta)*W1)
    rho = (mu0 - r)/sigma0 + delta*t
    mu = rho[None,:]*sigma + r
    
    dW = np.diff(W, axis=1, prepend=0) 
    integral_mu = np.cumsum((mu - 0.5*sigma**2)*dt, axis=1)
    integral_mu[:, 1:] = integral_mu[:, :-1]  
    integral_mu[:, 0] = 0
    integral_sigma = np.cumsum(sigma*dW, axis=1)
    S = np.exp(integral_mu + integral_sigma)
    market_non_statio = (S, t, S)
    return market_non_statio


##################################      EMV algo     ###########################################

def compute_V(t, x, w, theta, T):
    """
        Function that computes the value function
    """
    return (x-w)**2 * np.exp(-theta[3]*(T-t)) + theta[2]*t**2 + theta[1]*t + theta[0]


def compute_base(t, x, w, theta, phi, lambd, T):
    """
        Function that computes the base term appearing in all the gradient formulae. This is to avoid multiple computations
    """
    dt = t[1] - t[0]
    V_point = ( compute_V(t[1:], x[1:], w, theta, T) - compute_V(t[:-1], x[:-1], w, theta, T) ) / dt
    base = V_point - lambd * (phi[0] + phi[1]*(T-t[:-1]))
    return base


def compute_dC_theta1(t, base):
    """
        Gradient of C wrt theta1
    """
    dt = t[1] - t[0]
    dC_theta1 = dt*np.sum(base)
    return dC_theta1


def compute_dC_theta2(t, base):
    """
        Gradient of C wrt theta2
    """
    dC_theta2 = np.sum( base * ((t[1:])**2 - t[:-1]**2) )
    return dC_theta2


def compute_dC_phi1(t, base, lambd, clip_grad):
    """
        Gradient of C wrt phi1
    """
    dt = t[1] - t[0]
    dC_phi1 = -lambd*dt*np.sum(base)
    return np.clip(dC_phi1, -clip_grad, clip_grad)


def compute_dC_phi2(t, x, w, phi, base, lambd, T, clip_grad):
    """
        Gradient of C wrt phi2
    """
    dt = t[1] - t[0]
    term1 = (x[1:]-w)**2 * np.exp(-2*phi[1]*(T-t[1:])) * (T-t[1:])
    term2 = (x[:-1]-w)**2 * np.exp(-2*phi[1]*(T-t[:-1])) * (T-t[:-1])
    dC_phi2 = dt * np.sum( base * ( -(2*term1 - 2*term2)/dt - lambd*(T-t[:-1]) ) )
    return np.clip(dC_phi2, -clip_grad, clip_grad)


def update_theta_phi(theta, phi, eta_theta, eta_phi, dC_theta1, dC_theta2, dC_phi1, dC_phi2, w, z, T):
    """
        Updates the parameters 
    """
    theta[1] = theta[1] - eta_theta*dC_theta1
    theta[2] = theta[2] - eta_theta*dC_theta2
    theta[0] = -theta[2]*T**2 - theta[1]*T - (w-z)**2
    theta[3] = 2*phi[1]
    phi[0] = phi[0] - eta_phi*dC_phi1
    phi[1] = max(0, phi[1] - eta_phi*dC_phi2)
    return theta, phi


def moyenne(x, w, phi, lambd, rho):
    """
        Outputs the mean of the gaussian policy. Here, only the sign of the market parameter rho is used
    """
    moy = -np.sqrt(2*phi[1]/lambd/np.pi) * np.exp((2*phi[0]-1)/2) * (x-w)
    return moy if rho>0 else -moy


def variance(ti, phi, T):
    """
        Outputs the variance of the gaussian policy.
    """
    var = (1/2/np.pi)*np.exp(2*phi[1]*(T-ti) + 2*phi[0] - 1)
    return var


def sample_x_EMV(t, w, phi, lambd, T, x0, k, market, r, rho):
    """
        Samples the wealth process according to the gaussian policy. 
    """
    S = market[2][k]
    dt = t[1] - t[0]
    x = np.zeros_like(S)
    x[0] = x0
    for i in range(1, len(t)):
        moy = moyenne(x[i-1], w, phi, lambd, rho)
        var = variance(t[i-1], phi, T)
        u = rng.normal(loc=moy, scale=np.sqrt(var), size=moy.shape)
        x[i] = x[i-1] + u*(S[i]-S[i-1])/np.maximum(S[i-1], 1e-6) - u*r*dt
    return x


def EMV_algo(w_init, theta_init, phi_init, alpha, eta_theta, eta_phi, lambd0, T, market, z, x0, M, N, r, rho, 
             decay_E=False, beta=200, clip_grad=3.):
    """
        Implements the EMV algorithm (offline).
        Parameters:
        - decay_E (bool): If True, uses decaying exploration (default: exponential decay from the paper with parameter beta).
    """
    
    t = market[1]
    x_T = np.zeros(M)
    w, theta, phi = w_init, theta_init.copy(), phi_init.copy()
    lambd_dec1 = lambd0*(1 - np.exp( beta * (np.arange(M) - M) / M ))
    
    for k in range(M):
        
        if decay_E:
            lambd = lambd_dec1[k] 
        else:
            lambd = lambd0

        x = sample_x_EMV(t, w, phi, lambd, T, x0, k, market, r, rho)
        base = compute_base(t, x, w, theta, phi, lambd, T)
        dC_theta1 = compute_dC_theta1(t, base)
        dC_theta2 = compute_dC_theta2(t, base)
        dC_phi1 = compute_dC_phi1(t, base, lambd, clip_grad)
        dC_phi2 = compute_dC_phi2(t, x, w, phi, base, lambd, T, clip_grad)
        theta, phi = update_theta_phi(theta, phi, eta_theta, eta_phi, dC_theta1, dC_theta2, dC_phi1, dC_phi2, w, z, T)
        x_T[k] = x[-1]
        if (k+1)%N==0:
            w = w - alpha*(np.mean(x_T[k-N+1:k+1]) - z)
    return w, theta, phi, x_T


def backtest_EMV(market, w, phi, x0, lambd, T, r, rho, nb_sample=1000):
    t = market[0]
    S = market[1]
    dt = t[1] - t[0]
    x = np.zeros((nb_sample, len(t)))
    x[:, 0] = x0
    u_list = np.zeros((nb_sample, len(t)))
    for i in range(1, len(t)):
        moy = moyenne(x[:,i-1], w, phi, lambd, rho)
        var = variance(t[i-1], phi, T)
        u = rng.normal(loc=moy, scale=np.sqrt(var), size=moy.shape)
        u_list[:,i-1] = u
        x[:,i] = x[:,i-1] + u*(S[i]-S[i-1])/np.maximum(S[i-1], 1e-6) - u*r*dt
    u_list[:,i] = u
    return x, u_list




########################### Aternative 1 MLE  ########################################

def Estimator(t, S, start, end):
    """ 
        Estimates the drift and the volatility of the price under a stationnary market 
    """
    price = S[:,start:end+1]
    dt = t[1] - t[0]
    returns = np.diff(np.log(price), axis=1)
    mu_hat = np.mean(returns, axis=1)/dt + 0.5*np.var(returns, axis=1, ddof=0)/dt
    sigma_hat = np.std(returns, axis=1, ddof=0)/np.sqrt(dt)
    return mu_hat, sigma_hat
    
def MLE_algo(T, market, z, x0, M, r, nb=100):
    """
        Emplements the MLE algorithm
    """
    x_T = np.zeros(M)
    t = market[1]
    S_past = market[0]
    S = market[2]
    S_tot = np.concatenate((S_past, S), axis=1)
    end = S_past.shape[1]
    dt = t[1] - t[0]
    x = np.zeros_like(S)
    x[:,0] = x0
    
    u_list = []
    for i in range(1, len(t)):
        mu_hat, sigma_hat = Estimator(t, S_tot, end+i-nb, end+i-1)
        rho_hat = (mu_hat - r)/sigma_hat
        y = T * rho_hat**2
        exp_y = np.where(y < 1e-6, 1 + y + y**2/2 + y**3/6 + 1e-6, np.exp(y))
        w = ( z*exp_y - x0 ) / ( exp_y - 1 )
        u_prior = -(rho_hat/sigma_hat) * (x[:,i-1] - w)
        u = np.clip(u_prior, -np.percentile(u_prior, 25), np.percentile(u_prior, 75))
        u_list.append(u)
        x[:,i] = x[:,i-1] + u*(S[:,i]-S[:,i-1])/np.maximum(S[:,i-1], 1e-6) - u*r*dt
    x_T = x[:,-1]
    return x_T, np.array(u_list)


