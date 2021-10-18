#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 21:14:30 2021

@author: walt
"""

import numpy as np
import matplotlib.pyplot as plt

class Monte_Carlo_Guaranteed_Annuity_Option:
    
    def __init__(self, M, T, g, n, N):
        
        self.M = M # Number of Monte Carlo runs
        self.N = N
        self.T = T        
        self.dt = self.T / self.N
        self.n = n
        self.g = g        
        # list comprehension
        fmax = lambda x: np.array([max(i, 0) for i in x])
        
        self.sigma = np.array([0.0452, 0.0368, 0.0015])
        self.k = np.array([0.3731, 0.011, 0.01])
        self.theta = np.array([0.074484, 0.245455, 0.0013])
        self.x0 = np.array([0.0510234, 0.0890707, 0.0004])
        
        
        
        # generate dW and forward X process
        # dWi = (dW1i, dW2i,..., dWDi)^T
        self.dW = np.sqrt(self.dt) * np.random.randn(self.M, 3, self.N + 1)
        self.dW[:, :, self.N][:] = np.NaN # dW[N] is not defined since dW[N] = W[N+1] - w[N] and W[N+1] is NaN
        
        #print(dW)
        self.X = np.zeros((self.M, 3, self.N+1))
        self.X[:, :, 0] = self.x0
        
        for m in np.arange(self.M):
            for n in np.arange(self.N):
                self.X[m, :, n+1] = self.X[m, :, n] + self.k * (self.theta - fmax(self.X[m, :, n])) * self.dt + \
                      self.sigma * np.sqrt(fmax(self.X[m, :, n])) * self.dW[m, :, n]
                      
        #print(self.X)
        
#        plt.plot(self.X[0, 0, :], label = 'd1')
#        plt.plot(self.X[0, 1, :], label = 'd2')
        plt.plot(self.X[0, 2, :], label = 'd3')
        plt.legend()
        
        
        C50_15 = 0.014
        EX2 = self.x0[1] * np.exp(-self.k[1] * self.T) + self.theta[1] * (1.0 - np.exp(-self.k[1] * self.T))
        EX3 = self.x0[2] * np.exp(-self.k[2] * self.T) + self.theta[2] * (1.0 - np.exp(-self.k[2] * self.T))
        
        self.m2 = -0.3
        self.m3 = (C50_15 - self.m2 * EX2) / EX3
        
        #print(self.m3)
        self.Ri = np.array([1, 1, 0])
        self.Mi = np.array([0, self.m2, self.m3])
        
        #print(self.phi(0, self.R + self.M))
        #print(self.psi(0, self.R + self.M))
        self.rbar = -0.12332
        self.mubar = 0.0
        
        #print(self.SZCB_P(0, self.T, self.X[0, :, 0]))
        
    def Monte_Carlo_Price(self):
        monte_value = 0.0
        
        for m in np.arange(self.M):
            summSi = 0.0
            for i in np.arange(1, self.n):
                summSi += self.SZCB_P(self.T, self.T + i, self.X[m, :, self.N])
            payoff = max(summSi - 1 / self.g, 0.0)
            monte_value += payoff
            #print(summSi)
            #print(payoff)
        monte_price = self.g * self.SZCB_P(0, self.T, self.x0) * monte_value / self.M
        print(monte_price)
        #return(monte_price)
        
    def eta(self, u):
        return np.sqrt(self.k ** 2 + 2 * u * self.sigma ** 2)
    
    # solution 1 of Riccatti equations
    def psi(self, tau, u):
        return 2 * u / (self.eta(u) + self.k) \
           - 4 * u * self.eta(u) / ((self.eta(u) + self.k) * \
           ((self.eta(u) + self.k) * np.exp(self.eta(u) * tau) + self.eta(u) - self.k))
    
    # solution 2 of Riccatti equations
    def phi(self, tau, u):
        return -self.k * self.theta * (self.eta(u) + self.k) * tau / self.sigma ** 2\
            + 2 * self.k * self.theta * np.log((self.eta(u) + self.k) * np.exp(self.eta(u) * tau) + self.eta(u) - self.k) / self.sigma ** 2\
            - 2 * self.k * self.theta * np.log(2 * self.eta(u)) / self.sigma ** 2
    
    # price of the survival zero coupon bond
    def SZCB_P(self, t1, t2, Xt1):
        
        return np.exp(-(self.rbar + self.mubar) * (t2 - t1)) * \
            np.prod(np.exp(-self.phi(t2 - t1, self.Ri + self.Mi) - self.psi(t2 - t1, self.Ri + self.Mi) * Xt1))
        
    
gao = Monte_Carlo_Guaranteed_Annuity_Option(M = 1000, T = 15, g = 0.111, n = 35, N = 1000)   
gao.Monte_Carlo_Price()    