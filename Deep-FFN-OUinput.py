# coding: utf-8

from brian2 import *
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import time

seed(4321)
np.random.seed(4321)

# determine the save path
savePath = './data/deepFFN_OU'

if os.path.exists(savePath):
    warnings.warn('{} exists (possibly so do data).'.format(savePath))
else:
    os.makedirs(savePath)


# Simulate the FFN with given beta_w and synaptic conductance

def FFN_run(I_inj, specs):

    # simulation dt
    defaultclock.dt = 0.01 * ms
    deltatime = defaultclock.dt

    # Simulation parameters
    duration = 600 * ms

    # Network macro parameter

    n_groups = specs['nlayers']
    group_size_exc = specs['group_size']

    V_thresh = -10 * mV
    V_rest = -80 * mV

    D = 0 * ms

    # Biophysical parameters

    E_exc = 0 * mV

    # tau for double exponential for EPSC
    Tau_epsc_1 = 0.5 * ms
    Tau_epsc_2 = 4 * ms

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

    # array of spike index and time, the second dimension indicates the order of trial
    I_sp = np.zeros([500000, 1])
    T_sp = np.zeros([500000, 1])

    tauxi = 1.0 * ms  # noise time scale

    counter = 0

    start_scope()

    # M-L model
    eqs_exc = '''
        dv/dt = (-g_l*(v-E_l) - g_Na*m_inf*(v-E_Na) - g_K*w*(v-E_K) + g_syn*y2*(E_exc-v)/cm**2 + I_noise/cm**2 + I_ext/cm**2) / C : volt
        dw/dt = phi*(w_inf-w) / tau_w : 1
        m_inf = 0.5 * (1 + tanh( (v-V_1)/V_2 )) : 1
        w_inf = 0.5 * (1 + tanh( (v-V_3)/V_4 )) : 1
        tau_w = ms / cosh( (v-V_3)/(2*V_4) ) : second
        V_3 : volt
        dI_noise/dt = -I_noise/tauxi + sigmaV*xi*tauxi**-0.5 :amp
        I_ext : amp
        dy2/dt = -y2/Tau_epsc_2 + y/Tau_epsc_1 : 1
        dy/dt = -y/Tau_epsc_1: 1
        sigmaV : amp
        g_syn : siemens
    '''

    #############################Neuron group Defining##############################

    G_exc = NeuronGroup(group_size_exc * n_groups, eqs_exc, threshold='v>V_thresh', method='heun', refractory=3.3 * ms)
    G_exc.v = E_l  # initialize rest potential
    G_exc.w = 0  # initialize synaptic weight
    G_exc.g_syn = 0 * uS  # initialize synaptic conductance
    G_exc.I_ext = 0 * uA

    # define b_w(namely, V_3) of each layer
    for z in range(n_groups):
        if z % 2 == 0:  # L1,L3,...
            G_exc.V_3[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['DIFF']['betaw'] * mV
            G_exc.g_syn[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['DIFF'][
                                                                             'gsyn'] * uS  # the coefficient of EPSC receiption
            G_exc.sigmaV[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['DIFF'][
                'sigmaV']  # the coefficient of EPSC receiption
        if z % 2 == 1:  # L2,L4,...
            G_exc.V_3[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INT']['betaw'] * mV
            G_exc.g_syn[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INT'][
                                                                             'gsyn'] * uS  # the coefficient of EPSC receiption
            G_exc.sigmaV[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INT'][
                'sigmaV']  # the coefficient of EPSC receiption

    # input layer
    G_exc.V_3[0 * group_size_exc:(0 + 1) * group_size_exc:1] = specs['INPUT']['betaw'] * mV
    G_exc.g_syn[0 * group_size_exc:(0 + 1) * group_size_exc:1] = specs['INPUT'][
                                                                     'gsyn'] * uS  # the coefficient of EPSC receiption
    G_exc.sigmaV[0 * group_size_exc:(0 + 1) * group_size_exc:1] = specs['INPUT'][
        'sigmaV']  # the coefficient of EPSC receiption

    #############################inter-group synapses##############################

    S_e2e = Synapses(G_exc, G_exc, 'w_syn: 1', on_pre='y_post += w_syn')
    for z in range(n_groups - 1):
        if z > 0:
            if z % 2 == 0:  # L1->L2, L3->L4, ... This should be sparser than the other case
                S_e2e.connect(
                    condition='i>z*group_size_exc-1 and i<(z+1)*group_size_exc and j>(z+1)*group_size_exc-1 and j<(z+2)*group_size_exc',
                    p=specs['DIFF']['p'])
            if z % 2 == 1:  # L2->L3, L4->L5, ...
                S_e2e.connect(
                    condition='i>z*group_size_exc-1 and i<(z+1)*group_size_exc and j>(z+1)*group_size_exc-1 and j<(z+2)*group_size_exc',
                    p=specs['INT']['p'])

    S_e2e.connect(
        condition='i>0*group_size_exc-1 and i<(0+1)*group_size_exc and j>(0+1)*group_size_exc-1 and j<(0+2)*group_size_exc',
        p=specs['INPUT']['p'])

    S_e2e.delay = D  # fixed delay
    S_e2e.w_syn = 1  # ' 2 * rand()' # random synaptical weight with the mean of 1

    #############################OU input##############################
    tau_ou = 5 * ms
    sigma_ou = I_inj / 6
    mu_ou = sigma_ou * 2

    eqs_input = '''
    dv/dt = (mu_ou - v)/tau_ou + sigma_ou * xi * tau_ou**-0.5 : amp
    '''

    G_ou_input = NeuronGroup(1, eqs_input, threshold='True', method='heun', refractory=deltatime)
    G_ou_input.v = 0

    Sinput = Synapses(G_ou_input, G_exc[:group_size_exc], 'w_input : 1', on_pre='I_ext_post = w_input * v_pre')
    Sinput.connect(p=1)
    Sinput.w_input = 1
    Sinput.delay = 0.0 * ms

    st_ou_input = StateMonitor(G_ou_input, 'v', record=range(1))
    Sp_exc = SpikeMonitor(G_exc)

    #############################Run and plot##############################
    seed(1212)
    np.random.seed(1212)

    run(duration)

    #######################formatize data#########################

    tmp_t = Sp_exc.t / ms
    tmp_i = array(Sp_exc.i) + 1  # due to that matlab index start from 1
    spike_num = size(Sp_exc.t)
    input_current = st_ou_input.v / uA
    T_sp[0:spike_num:1, counter] = tmp_t.reshape([spike_num])
    I_sp[0:spike_num:1, counter] = tmp_i.reshape([spike_num])

    counter += 1
    print(counter)

    data = {'groupsize': group_size_exc,
            'ngroups': n_groups, 'spikei': I_sp, 'spiketime': T_sp, 'duration': duration / ms,
            'Kinh': 0,
            'TauSynapse': [Tau_epsc_1, Tau_epsc_2], 'DT': defaultclock.dt / ms,
            'input': input_current
            }
    return (data)


# # Heterogeneous FFN

group_size = 1000
pbase = 9.0 / group_size
convergence = 9.0 / 9.0

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA', 'p': pbase}
         }

data = FFN_run(75 * uA, specs)

# st = data['spiketime'][:, 0]
# si = data['spikei'][:, 0]
#
# si = si[st > 0]
# st = st[st > 0]
#
# _, ax = plt.subplots(figsize=(20, 5))
# ax.plot(st, si, '.', markersize=1)
#
# _, ax = plt.subplots(nrows=7, figsize=(20, 5), sharex=True)
# for z, _ in enumerate(ax):
#     _ = ax[z].hist(st[np.logical_and(si >= z * group_size, si < (z + 1) * group_size)], np.arange(0, 600))

sio.savemat(savePath + '/HetFFN_5ms_75uA_600ms.mat', data)

# # CD FFN

convergence = 1.0
specs = {'nlayers': 7,
         'group_size': group_size,
         'INT': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA', 'p': pbase}
         }

data = FFN_run(75 * uA, specs)

# st = data['spiketime'][:, 0]
# si = data['spikei'][:, 0]
#
# si = si[st > 0]
# st = st[st > 0]
#
# _, ax = plt.subplots(figsize=(20, 5))
# ax.plot(st, si, '.', markersize=1)
#
# _, ax = plt.subplots(nrows=7, figsize=(20, 5), sharex=True)
# for z, _ in enumerate(ax):
#     _ = ax[z].hist(st[np.logical_and(si >= z * group_size, si < (z + 1) * group_size)], np.arange(0, 600))

sio.savemat(savePath + '/CDFFN_5ms_75uA_600ms.mat', data)

# # INT FFN

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA', 'p': pbase}
         }

data = FFN_run(75 * uA, specs)

# st = data['spiketime'][:, 0]
# si = data['spikei'][:, 0]
#
# si = si[st > 0]
# st = st[st > 0]
#
# _, ax = plt.subplots(figsize=(20, 5))
# ax.plot(st, si, '.', markersize=1)
#
# _, ax = plt.subplots(nrows=7, figsize=(20, 5), sharex=True)
# for z, _ in enumerate(ax):
#     _ = ax[z].hist(st[np.logical_and(si >= z * group_size, si < (z + 1) * group_size)], np.arange(0, 500))

sio.savemat(savePath + '/INTFFN_5ms_75uA_600ms.mat', data)
