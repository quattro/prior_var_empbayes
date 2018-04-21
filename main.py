#! /usr/bin/env python
from __future__ import print_function

import argparse as ap
import random as rdm
import os
import sys

import numpy as np
import numpy.linalg as lin
import scipy.stats as stats
import scipy.optimize as opt

from numpy.linalg import multi_dot


class Func(object):

    def __init__(self):
        pass

    def func(self, p):
        pass

    def grad(self, p):
        pass

    def hess(self, p):
        pass


class LBF(Func):

    def __init__(self, N, K, prior_var, seed=None):
        rdm.seed(seed)
        V = stats.wishart.rvs(N, np.eye(K))
        D = np.diag(np.ones(K) * prior_var)
        S = prior_var * V.dot(V) + V

        Z = stats.multivariate_normal.rvs(size=1, cov=S)
        eigs, U = lin.eig(V)

        self.Zsq = Z.dot(U) ** 2
        self.D = eigs

        return

    def func(self, p):
        term1 = 0.5 * -np.sum(np.log(1 + p * self.D)) 
        term2 = 0.5 * np.sum((self.Zsq * p) / (1 + p * self.D))  
        return term1 + term2

    def grad(self, p):
        term1 = 0.5 * -np.sum(self.D / (1 + p * self.D)) 
        term2 = 0.5 * np.sum(self.Zsq / ((1 + p * self.D) ** 2))
        return term1 + term2

    def hess(self, p):
        term1 = 0.5 * np.sum(self.D**2 / ((1 + p * self.D)**2))
        term2 = -np.sum((self.D * self.Zsq) / ((1 + p * self.D)**3))
        return term1 + term2


class LMVN(Func):

    def __init__(self, N, K, prior_var, seed=None):
        rdm.seed(seed)
        V = stats.wishart.rvs(N, np.eye(K))
        D = np.diag(np.ones(K) * prior_var)
        S = prior_var * V.dot(V) + V

        self.Z = stats.multivariate_normal.rvs(size=1, cov=S)
        self.V = V
        self.V2 = V.dot(V)

        return

    def func(self, p):
        S = p * self.V2 + self.V
        return stats.multivariate_normal.logpdf(self.Z, cov=S)

    def grad(self, p):
        S = p * self.V2 + self.V
        Vinv = lin.pinv(S)
        VinvVV = Vinv.dot(self.V2) 
        return -0.5 * np.trace(VinvVV) + 0.5 * multi_dot([self.Z, VinvVV, Vinv, self.Z])

    def hess(self, p):
        S = p * self.V2 + self.V
        Vinv = lin.pinv(S)
        VinvVV = Vinv.dot(self.V2)
        VinvVV2 = VinvVV.dot(VinvVV)
        return 0.5 * np.trace(VinvVV2) - 0.5 * multi_dot([self.Z, VinvVV2, Vinv, self.Z])


def comp_func(funcs, negate=True):
    if negate:
        return lambda x: -sum(f.func(x) for f in funcs)
    else:
        return lambda x: sum(f.func(x) for f in funcs)


def comp_grad(funcs, negate=True):
    if negate:
        return lambda x: -sum(f.grad(x) for f in funcs)
    else:
        return lambda x: sum(f.grad(x) for f in funcs)


def comp_hess(funcs, negate=True):
    if negate:
        return lambda x: -sum(f.hess(x) for f in funcs)
    else:
        return lambda x: sum(f.hess(x) for f in funcs)


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)

    K = 300
    N = 1000
    prior_var = 50

    lmvns = [LMVN(N, K, prior_var, seed=777), LMVN(N, K, prior_var)]
    lbfs  = [LBF(N, K, prior_var, seed=777), LBF(N, K, prior_var)]

    lmvn_f = comp_func(lmvns)
    lmvn_g = comp_grad(lmvns)
    lmvn_h = comp_hess(lmvns)

    lbf_f = comp_func(lbfs)
    lbf_g = comp_grad(lbfs)
    lbf_h = comp_hess(lbfs)

    callback = lambda x: print("Current val = {}".format(x))

    p0 = 10
    print("Optimizing MVN")
    res = opt.fmin_ncg(lmvn_f, p0, fprime=lmvn_g, fhess=lmvn_h, callback=callback)

    print("Optimizing LBF")
    res = opt.fmin_ncg(lbf_f, p0, fprime=lbf_g, fhess=lbf_h, callback=callback)


    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
