#!/usr/bin/python
# -*- coding: UTF-8 -*-
from matplotlib import *
import numpy as n
def cofiCostFunc(params,Y,R,num_users,num_movies,num_features,lamb):
    X = n.reshape(params[0:num_movies*num_features],(num_movies,num_features),order='F')
    Theta = n.reshape(params[num_movies*num_features:],(num_users,num_features),order='F')
    err = (X.dot(Theta.T)) - Y
    M = err*err
    reg = (n.sum(n.sum(Theta**2,axis=1))+n.sum(n.sum(X**2,axis=1)))*(lamb/2)

    J = n.sum(n.sum(R*M))/2+reg
    X_grad = (err*R).dot(Theta)+X.dot(lamb)
    Theta_grad = (err*R).T.dot(X) + Theta.dot(lamb)
    grad = n.r_[X_grad.flatten(1),Theta_grad.flatten(1)]
    grad = n.reshape(grad,(-1,1),order = 'F')
    return (J,grad)
