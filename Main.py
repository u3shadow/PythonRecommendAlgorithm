#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as n
import scipy.io as io

from CostFunctionFi import cofiCostFunc

params = io.loadmat('ex8_movieParams.mat')
print params.keys()
movies = io.loadmat('ex8_movies.mat')
print movies.keys()
X = params['X']
Y = movies['Y']
R = movies['R']
Theta = params['Theta']
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies][:,0:num_features]
Theta = Theta[0:num_users][:,0:num_features]
Y = Y[0:num_movies][:,0:num_users]
R = R[0:num_movies][:,0:num_users]
J = cofiCostFunc(n.reshape(n.r_[X.flatten(1),Theta.flatten(1)],(-1,1),'F'),Y,R,num_users,num_movies,num_features,1.5)
print J[0]

