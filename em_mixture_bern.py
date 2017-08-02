# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 20:39:46 2017

@author: zx_pe
"""
import numpy as np
import cleaner
from cleaner import PreprocessBinaryData

class em_mixture_bernoulli:
    def __init__(self):
        self.ll = -10000
        self.lws = 0
        self.gammas = 0
        self.mus = 0

    def logsumexp(self,x):
        return np.log(sum(np.exp(x))+1e-50)
    
    def pbern(self,x,u):
        a = np.array(x)
        b = np.array(u)
        return np.prod((b**a) * ((1-b)**(1-a)))
        
    def loglik(self,x, mus, lws):
        n = np.shape(x)[0]
        k = np.shape(mus)[0]
        ll = 0
        for i in range(n):
            s = 0
            for j in range(k):
                s += np.exp(lws[j]) * self.pbern(x[i,],mus[j,])
            s += 1e-50
            ll += np.log(s)
        if np.isnan(ll):
            ll = 1e5
        return ll
        
    def commonWords(self, clusterN, n = 10):
        b = self.labels == clusterN
        tweets = self.raw.iloc[b,:]
        wordCount = tweets.sum(0)
        wordCount.sort(ascending = 0)
        return wordCount[0:n]

    def wordCount(self, clusterN):
        b = self.labels == clusterN
        tweets = self.raw.iloc[b,:]
        wordCount = tweets.sum(0)
        return wordCount
        
    def loadData(self, x, min_count):
        p = cleaner.PreprocessBinaryData(x, min_count)
        self.raw = p
        self.xs = np.matrix(p)
        self.n, self.m = np.shape(self.xs)
        self.labels = list(np.repeat(0,self.n))
        
    def train(self, k, max_it = 9999):
        self.lws = np.repeat(np.log(1/(k+1e-50)),k)
        self.gammas = np.zeros([self.n,k])
        self.mus = np.random.random((k,self.m))
        converged = False
        numit = 0
        self.ll = -10000
        while(numit < max_it and not converged):
            numit += 1
            mus_old = self.mus
            ll_old = self.ll
            print("Iteration {}, log-likelihood: {}".format(numit, self.ll))
            
            #E-step
            for i in range(self.n):
                lprs = [self.lws[j] + np.log(max(0, 1e-50+self.pbern(self.xs[i,],self.mus[j,]))) for j in range(k)]
                self.gammas[i,] = np.exp(lprs - self.logsumexp(lprs))
            self.ll = self.loglik(self.xs,self.mus,self.lws)
            
            #M-step - update weights, mus
            Ns = np.repeat(0.0,k)
            for j in range(k):
                Ns[j] = sum(self.gammas[:,j])
                self.lws[j] = np.log(Ns[j]) - np.log(self.n)
                self.mus[j,] = sum([self.gammas[i,j]/Ns[j]*self.xs[i,] for i in range(self.n)])
            if np.abs(self.ll - ll_old) < 1e-5:
                converged = True
        self.labels = np.array([np.argmax(i) for i in self.gammas])
