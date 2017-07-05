import numpy as n
def computeNumbericalGradient(J,theta):
    numgrad = n.zeros(theta.shape)
    perturb = n.zeros(theta.shape)
    e = pow(10,-2)
    for p in range(0,theta.size):
        perturb[p] = e
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta +perturb)[0]
        numgrad[p] = (loss2 - loss1) /(2*e)
        perturb[p] = 0
    return numgrad
