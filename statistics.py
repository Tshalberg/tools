from __future__ import division
import numpy as np
from numpy import array
import numpy.random as r
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fmin

def get_ymax(f, params, xbounds, N=1000, delta_y=0.01):
    dN = 1./N
    xmin, xmax = xbounds
    x = np.arange(xmin, xmax+dN, dN)
    y = f(x, params)
    return max(y)*(1+delta_y)

def sample(f, params, xbounds, Naccepted=2000, full_output=0):

    xmin, xmax = xbounds
    ymin, ymax = 0, get_ymax(f, params, xbounds)
    
    accepted = []
    rejected = []
    while len(accepted) < Naccepted:
        x = r.uniform(xmin, xmax)
        y = r.uniform(ymin, ymax)
        if y <= f(x, params):
            accepted.append([x, y])
        else:
            rejected.append([x, y])
    accepted = array(accepted)
    rejected = array(rejected)
    
    x = accepted[:,0]
    if full_output:
        return x, accepted, rejected
    else:
        return x

def fit_params(x, f, x0, N=2000, delta_n=0.0, full_output=1, retall=0, disp=0):

    def func(params):
        alpha, beta = params 
        val = np.sum(-np.log(f(x, params)))
        noise = r.randn() * delta_n
        return val + noise
    if delta_n <= 0.0:
        ftol = 0.001
    else:
        ftol = delta_n
    minObj = fmin(func, x0, ftol=delta_n, full_output=full_output, retall=retall, disp=disp)
    # print minObj
    if retall:
        alpha, beta = minObj[0]
        steps = minObj[-1]
        return alpha, beta, steps
    else:
        alpha, beta = minObj[0]
        warn = minObj[4]
        return alpha, beta, warn


def plot_resolutions(As, Bs, params, figname, bins=20):
    from tools.plotting import plot_bands, calc_bands
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    hist1 = ax1.hist(As, bins=bins)
    ax1.axvline(params[0], 0, 1, ls="-.", c="r", label="true: %.5s" % params[0], zorder=3)
    plot_bands(As, ax1)
    Abands = calc_bands(As)
    ax1.set_title("Resolution: %.5s" % (Abands[2] - Abands[0]))
    ax1.set_xlabel(r"$\alpha$", size=15)
    ax1.legend()

    hist2 = plt.hist(Bs, bins=bins)
    plot_bands(Bs, ax2)
    ax2.axvline(params[1], 0, 1, ls="-.", c="r", label="true: %.5s" % params[1], zorder=3)
    Bbands = calc_bands(Bs)
    ax2.set_title("Resolution: %.5s" % (Bbands[2] - Bbands[0]))
    ax2.set_xlabel(r"$\beta$", size=15)
    ax2.legend()

    plt.savefig("plots/statistics/resolutions/" + figname, dpi=300)

######

def poisson_llh(x, mu, deltamu=1e-1, vectorCalc=True):
    from scipy.special import loggamma as lgamma
    llh = 0
    if vectorCalc:
        mask_mu = mu != 0
        llh += np.sum(x[mask_mu]*np.log(mu[mask_mu]) - mu[mask_mu] - lgamma(x[mask_mu] + 1))
        llh += np.sum(x[~mask_mu] * np.log(deltamu) - mu[~mask_mu] - lgamma(x[~mask_mu] + 1))
        return llh
    else:
        for i in range(len(x)):
            if mu[i] > 0:
                llh += x[i] * np.log(mu[i]) - mu[i] - lgamma(x[i] + 1)
            else:
                llh += x[i] * np.log(deltamu) - mu[i] - lgamma(x[i] + 1)
        return llh


def LLH_dima(x, mu, os, deltamu=1e-1, vectorCalc=True) :
    from scipy.special import loggamma as lgamma
    x = x.copy()
    mu = mu.copy()
    '''
    Calculate dima LLH
    '''
    llh = 0
    if vectorCalc:
        mu_dima = (os*mu+x)/(os+1)
        mask_mu = mu != 0
        mask_x = x != 0
        llh += np.sum(os*mu[mask_mu]*np.log(mu_dima[mask_mu]/mu[mask_mu]))
        llh += np.sum(x[mask_x]*np.log(mu_dima[mask_x]/x[mask_x]))
        llh += np.sum(x[~mask_mu] * np.log(deltamu) - mu[~mask_mu] - lgamma(x[~mask_mu] + 1))
    else:
        for i in range(len(x)):
            mu_dima = (os*mu[i]+x[i])/(os+1)
            if mu[i] != 0:
                llh += os*mu[i]*np.log(mu_dima/mu[i])
            
            if x[i] != 0:
                llh += x[i]*np.log(mu_dima/x[i])

    return -llh


#     minObj = minimize(f,x0, bounds=bnds, method="nelder-mead", callback=callback)
#     alpha, beta = minObj.x

    # def func(x, params, c):
    #     return f(x, params)*c
# def constants(f, params, xbounds, N=10000, delta_y=0.01):
#     dN = 1./N
#     xmin, xmax = xbounds
#     x = np.arange(xmin, xmax+dN, dN)
#     y = f(x, params)
#     c = 1/(max(y)*dN)
#     return c, min(y)*(1-delta_y), max(y)*(1+delta_y)