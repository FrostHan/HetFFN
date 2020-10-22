from brian2 import *
import numpy as np


def exp2_factor(tau1, tau2):
    tp = (tau1 * tau2) / (tau2 - tau1) * np.log(tau2 / tau1)
    factor = -np.exp(-tp / tau1) + np.exp(-tp / tau2)
    return 1.0 / factor


def density(data, t, Nt, duration):
    tt = linspace(0.0, duration / ms, Nt)
    dtt = tt[1] - tt[0]
    den = zeros([Nt, 1])
    ST = size(t)
    j = 0
    for i in range(Nt):
        while logical_and(abs(t[j] / ms - tt[i]) < dtt / 2, j < ST - 1):
            den[i, 0] += 1
            j += 1
    return den, tt  # tt dimensionless


def std_spiketime(Sp, duration, range_i):
    # Compute the standard deviation of spike time, which indicate the ability of synchrony signal propagation
    data0 = Sp.i
    t0 = Sp.t
    ind1 = np.where(logical_and(range_i[0] <= data0, data0 <= range_i[-1]))

    a1 = ind1[0]

    data = data0[a1]
    t = t0[a1]

    den, tt = density(data, t, 200, duration)

    ind_center = argmax(den)
    t_center = tt[ind_center] * ms

    inds_in = np.where(abs((t - t_center) / ms) < 4.0)

    a = inds_in[0]
    t_in = t[a]

    sig = std(t_in)
    return sig


def firing_rate(Sp, duration, range_i):
    data0 = Sp.i
    t0 = Sp.t
    ind1 = np.where(logical_and(range_i[0] <= data0, data0 <= range_i[-1]))

    a1 = ind1[0]

    data = data0[a1]
    t = t0[a1]

    den, tt = density(data, t, 200, duration)

    ind_center = argmax(den)
    t_center = tt[ind_center] * ms

    inds_in = np.where(abs((t - t_center) / ms) < 4.0)

    a = inds_in[0]
    t_in = t[a]

    fr = size(t_in)
    return fr
