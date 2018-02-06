from __future__ import absolute_import
from __future__ import print_function
from .utils import relative
from .utils import witness
import numpy as np


def rlt(X, L_0=64, gamma=0.01, i_max=100):
    # This function implements Algorithm 1 from the paper
    # for an individual sample of landmarks
    I_1, alpha_max = witness(X, L_0=L_0, gamma=gamma)
    res = relative(I_1, alpha_max, i_max=i_max)
    return res


def rlts(X, L_0=64, gamma=0.01, i_max=100, n=1000):
    # This function implements Algorithm 1 from the paper
    rlts = np.zeros((n, i_max))
    for i in range(n):
        rlts[i, :] = rlt(X, L_0, gamma, i_max)
        if i % 10 == 0:
            print(i)
    return rlts


def geom_score(rlts1, rlts2):
    # This function implements Algorithm 2 from the paper
    mrlt1 = np.mean(rlts1, axis=0)
    mrlt2 = np.mean(rlts2, axis=0)
    return np.sum((mrlt1 - mrlt2) ** 2)
