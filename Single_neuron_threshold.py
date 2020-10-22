# coding: utf-8

from brian2 import *
from utils import *
import numpy as np
import scipy.io as sio
import os, time, warnings

this_seed = 4321
seed(this_seed)
np.random.seed(this_seed)

savePath = './data'
if os.path.exists(savePath):
    warnings.warn('{} exists (possibly so do data).'.format(savePath))
else:
    os.makedirs(savePath)

def run1(Iinj, sigmaI, betaw):
    start_scope()

    # Simulation parameters
    defaultclock.dt = 0.1 * ms
    duration = 400 * ms

    sigma = 0.0 * ms
    D = 0 * ms  # Synaptic delay

    # Cell parameters
    E_Na = 50 * mV
    E_K = -100 * mV
    E_l = -70 * mV
    g_Na = 20 * msiemens / cm ** 2
    g_K = 20 * msiemens / cm ** 2
    g_l = 2 * msiemens / cm ** 2
    phi = 0.15
    C = 2 * ufarad / cm ** 2
    V_1 = -1.2 * mV
    V_2 = 18 * mV
    V_4 = 10 * mV

    sigmaV = sigmaI * uA  # noise level
    tauxi = 1.0 * ms  # noise time scale

    beta_w = betaw * mV
    if betaw < -20:
        Ib = 0 * uA
    else:
        Ib = 0 * uA

    I_inj = Iinj * uA

    #############################model equations##############################

    # conductance-based input
    eqs = '''
    dv/dt = (-g_l*(v-E_l) - g_Na*m_inf*(v-E_Na) - g_K*w*(v-E_K) + I_syn/cm**2 + I_noise/cm**2) / C : volt
    dw/dt = phi*(w_inf-w) / tau_w : 1
    m_inf = 0.5 * (1 + tanh( (v-V_1)/V_2 )) : 1
    w_inf = 0.5 * (1 + tanh( (v-V_3)/V_4 )) : 1
    tau_w = ms / cosh( (v-V_3)/(2*V_4) ) : second
    dI_noise/dt = -I_noise/tauxi + sigmaV*fnoise*xi*tauxi**-0.5 :amp
    I_syn : amp
    fnoise : 1
    V_3 : volt
    '''

    #############################Neuron group Defining##############################
    G = NeuronGroup(1, eqs, method='heun', threshold='v > -10 *mV', refractory=1 * ms)
    G.v = E_l
    G.w = 0
    G.I_syn = 0 * pA
    G.V_3 = beta_w

    ############################# Trigger ##############################

    Ginput = SpikeGeneratorGroup(1, [0], [50.0 * ms])  # trigger at 10ms
    Sinput = Synapses(Ginput, G,
                      on_pre='''
                      I_syn += I_inj
                      fnoise += 1
                      ''')

    Sinput.connect(j='i')
    Sinput.delay = D

    #############################Recoding##############################


    St = StateMonitor(G, 'v', record=range(1))
    Si1 = StateMonitor(G, 'I_noise', record=range(1))
    Si2 = StateMonitor(G, 'I_syn', record=range(1))
    Sp = SpikeMonitor(G)

    #############################Run and plot##############################
    seed(this_seed)
    np.random.seed(this_seed)
    run(duration)

    ############################record result#########################

    return St, Sp, Si1, Si2


def ML_single_run(Iinjs, sigmaI, beta_w):
    N_plot = Iinjs.size

    fre = zeros([N_plot])
    mp = zeros([4000, N_plot])
    irec = zeros([4000, N_plot])

    counter = 0

    for counter, I_inj in enumerate(Iinjs):
        St, Sp, Si1, Si2 = run1(I_inj, sigmaI, beta_w);
        print("progress: {}/{}".format(counter + 1, N_plot))
        fre[counter] = size(Sp.t / ms)
        mp[:, counter] = St.v / mV
        irec[:, counter] = Si1.I_noise / uA + Si2.I_syn / uA

    t = St.t / ms

    return t, mp, fre, Iinjs, irec


import h5py

f = h5py.File('single_cell_data.h5', 'w')
f.create_group('INT')
g = f.create_group('INT/no_noise')

def save_data():
    g['t'] = t
    g['mp'] = mp
    g['fre'] = fre
    g['Iinjs'] = Iinjs
    g['irec'] = irec
    f.flush()

icur = 40
_, ax = plt.subplots(nrows=2)
# t, mp, fre, Iinjs = ML_single_run(arange(-20,60,10), 0)
t, mp, fre, Iinjs, irec = ML_single_run(arange(icur, icur + 10, 10), 0.0, 0)
save_data()

g = f.create_group('INT/noise')
ax[0].plot(t, mp)
t, mp, fre, Iinjs, irec = ML_single_run(arange(icur, icur + 10, 10), 12.5, 0)
ax[1].plot(t, mp)
save_data()

f.create_group('DIFF')
g = f.create_group('DIFF/no_noise')

icur = 55
_, ax = plt.subplots(nrows=2)
# t, mp, fre, Iinjs = ML_single_run(arange(-20,60,10), 0)
t, mp, fre, Iinjs, irec = ML_single_run(arange(icur, icur + 10, 10), 0.0, -19)
ax[0].plot(t, mp)
save_data()

g = f.create_group('DIFF/noise')
t, mp, fre, Iinjs, irec = ML_single_run(arange(icur, icur + 10, 10), 12.5, -19)
ax[1].plot(t, mp)
save_data()
plt.show()

f.flush()
f.close()


f = h5py.File(savePath + '/single_cell_data.h5', 'r')

mp = f['INT/noise/mp']
_, ax = plt.subplots()
ax.plot(mp)

mp = f['DIFF/noise/mp']
_, ax = plt.subplots()
ax.plot(mp)

plt.show()

f.close()

import datetime

datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
