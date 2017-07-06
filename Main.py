#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as n
import scipy.io as io
import scipy as sci

from CheckCostFun import CheckCost
from CostFunctionFi import cofiCostFunc
from CostFunctionFiJ import cofiCostFuncJ
from CostFunctionFiG import cofiCostFuncG
from LoadMovieList import loadMovieList

params = io.loadmat('ex8_movieParams.mat')
movies = io.loadmat('ex8_movies.mat')
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
CheckCost(1.5)

movieList = loadMovieList()
my_ratings = n.zeros((1682,1))
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print "New user rattings:"

for i in range(0,my_ratings.size):
    if my_ratings[i] > 0:
        print "Rated %d for %s"%(my_ratings[i],movieList[i])

movies = io.loadmat('ex8_movies.mat')
Y = movies['Y']
R = movies['R']
Y = n.column_stack((my_ratings,Y))
R = n.column_stack((my_ratings,R))
Yshape = Y.shape
Ymean = n.zeros((Yshape[0],1))
Ynorm = n.zeros(Yshape)
for i in range(0,Yshape[0]):
    idx = n.nonzero(R[i,:])
    Ymean[i] = n.mean(Y[i,idx])
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]
num_users =  Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
X = n.random.rand(num_movies,num_features)
Theta = n.random.rand(num_users,num_features)

initial_parameters =n.reshape (n.r_[X.flatten(1),Theta.flatten(1)],(-1,1),'F')
lambd = 10
theta = sci.optimize.fmin(lambda t : cofiCostFuncJ(t,Y,R,num_users,num_movies,num_features,lambd),initial_parameters)
X = n.reshape(theta[0:num_movies*num_features],(num_movies,num_features),'F')
Theta = n.reshape(theta[num_movies*num_features:],(num_users,num_features),'F')
p = X.dot(Theta.T)
my_predictions = p[:,1]+Ymean
print "learning completed"


