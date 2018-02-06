import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import gudhi
import matplotlib.pyplot as plt


def relative(I_1, alpha_max, i_max=100):
    # Given a list of intervals I_1 this function computes RLT
    # using formulas (2) and (3) from the paper

    persistence_intervals = []
    for interval in I_1:
        if not np.isinf(interval[1]):
            persistence_intervals.append(list(interval))
        elif np.isinf(interval[1]):
            persistence_intervals.append([interval[0], alpha_max])

    if len(persistence_intervals) == 0:
        rlt = np.zeros(i_max)
        rlt[0] = 1.0
        return rlt

    persistence_intervals_ext = persistence_intervals + [[0, alpha_max]]
    persistence_intervals_ext = np.array(persistence_intervals_ext)
    persistence_intervals = np.array(persistence_intervals)

    switch_points = np.sort(np.unique(persistence_intervals_ext.flatten()))
    rlt = np.zeros(i_max)
    for i in range(switch_points.shape[0] - 1):
        midpoint = (switch_points[i] + switch_points[i + 1]) / 2
        s = 0
        for interval in persistence_intervals:
            if midpoint >= interval[0] and midpoint < interval[1]:
                s = s + 1
        if (s < i_max):
            rlt[s] += (switch_points[i + 1] - switch_points[i])

    return rlt / alpha_max


def lmrk_table(W, L):
    # Helper function to construct an input
    # for gudhi.WitnessComplex

    a = cdist(W, L)
    max_val = np.max(a)
    idx = np.argsort(a)
    b = a[np.arange(np.shape(a)[0])[:, np.newaxis], idx]
    return np.dstack([idx, b]), max_val


def random_landmarks(X, num_landmarks=32):
    sz = X.shape[0]
    idx = np.random.choice(sz, num_landmarks)
    L = X[idx]
    return L


def witness(X, gamma=1.0 / 128, L_0=64):
    L = random_landmarks(X, L_0)
    W = X
    lmrk_tab, max_dist = lmrk_table(W, L)
    wc = gudhi.WitnessComplex(lmrk_tab)
    alpha_max = max_dist * gamma
    st = wc.create_simplex_tree(max_alpha_square=alpha_max, limit_dimension=2)
    big_diag = st.persistence(homology_coeff_field=2)
    diag = st.persistence_intervals_in_dimension(1)
    return diag, alpha_max


def fancy_plot(y, color='C0', label='', alpha=0.3):
    n = y.shape[0]
    x = np.arange(n)
    xleft = x - 0.5
    xright = x + 0.5
    X = np.array([xleft, xright]).T.flatten()
    Xn = np.zeros(X.shape[0] + 2)
    Xn[1:-1] = X
    Xn[0] = -0.5
    Xn[-1] = n - 0.5
    Y = np.array([y, y]).T.flatten()
    Yn = np.zeros(Y.shape[0] + 2)
    Yn[1:-1] = Y
    plt.bar(x, y, width=1, alpha=alpha, color=color, edgecolor=color)
    plt.plot(Xn, Yn, c=color, label=label, lw=3)
