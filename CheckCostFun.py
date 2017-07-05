#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as n
import scipy as sci

from ComputeNumbericalGradient import computeNumbericalGradient
from CostFunctionFi import cofiCostFunc


def CheckCost(lambd):
    if lambd is None:
        lambd = 0
    X_t = n.random.rand(4,3)
    Theta_t = n.random.rand(5,3)
    Y = X_t.dot(Theta_t.T)
    Y[ n.random.rand(Y[:,1].size,Y[1,:].size) > 0.5] = 0
    R = n.zeros(Y.shape)

    for i in range(0,Y[:,1].size):
        for j in range(0,Y[1,:].size):
            if Y[i,j] != 0:
                R[i,j] = 1

    X = n.random.rand(4,3)
    Theta = n.random.rand(5,3)
    num_users = n.size(Y,1)
    num_movies = n.size(Y,0)
    num_features = n.size(Theta_t,1)
    J = cofiCostFunc(n.reshape(n.r_[X.flatten(1),Theta.flatten(1)],(-1,1),'F'),Y,R,num_users,num_movies,num_features,lambd)
    grad = J[1]
    numgrad = computeNumbericalGradient(lambda t:cofiCostFunc(t,Y,R,num_users,num_movies,num_features,lambd),n.reshape(n.r_[X.flatten(1),Theta.flatten(1)],(-1,1),'F'))
    diff =n.linalg.norm(numgrad -grad)/n.linalg.norm(numgrad+grad)
    print "less than le-9 %.10f"%diff
