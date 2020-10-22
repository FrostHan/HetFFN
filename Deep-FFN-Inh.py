# coding: utf-8

from brian2 import *
try:
    from python_utils import *
except:
    try:
        from utils import *
    except:
        pass
import numpy as np
import scipy.io as sio
import os, time, warnings


this_seed = 4321
seed(this_seed)
np.random.seed(this_seed)

# Determine where the .mat data saved
savePath =  './data/deepFFN-Inh'
if os.path.exists(savePath):
    warnings.warn('{} exists (possibly so do data).'.format(savePath))
else:
    os.makedirs(savePath)

inh_betaw_list = [-15.0,] # mV
g_inh_list = [200.0,] # uS

g_inh = 200.0
group_size_inh = 1000
betaw_inh = -15.0

# Simulate the FFN with given beta_w and synaptic conductance
def FFN_MP_run(sigma_list, alpha_list, specs):
    # simulation dt
    defaultclock.dt = 0.01 * ms

    # Simulation parameters
    duration = 180 * ms
    sigmas = sigma_list * ms  # sigma of input stimuli
    Nspikes = alpha_list  # alpha of input stimuli

    # Network macro parameter

    n_groups = specs['nlayers']
    group_size_exc = specs['group_size_exc']
    group_size_inh = specs['group_size_inh']

    #     p_connect = 0.01
    V_thresh = -10 * mV
    V_rest = -80 * mV

    D = 2 * ms  # delay
    D_inh = 2 * ms

    # Biophysical parameters

    E_exc = 0 * mV
    E_inh = -90 * mV

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
    I_sp_exc = np.zeros([500000, size(sigmas) * size(Nspikes)])
    T_sp_exc = np.zeros([500000, size(sigmas) * size(Nspikes)])

    I_sp_inh = np.zeros([200000, size(sigmas) * size(Nspikes)])
    T_sp_inh = np.zeros([200000, size(sigmas) * size(Nspikes)])

    tauxi = 1.0 * ms  # noise time scale

    counter = 0

    for sigma in sigmas:
        for n_spikes in Nspikes:

            n_spikes = int(n_spikes)

            start_scope()

            # M-L model
            eqs = '''
                dv/dt = (-g_l*(v-E_l) - g_Na*m_inf*(v-E_Na) - g_K*w*(v-E_K) + g_syn_exc*y2_exc*(E_exc-v)/cm**2 + g_syn_inh*y2_inh*(E_inh-v)/cm**2 + I_noise/cm**2) / C : volt
                dw/dt = phi*(w_inf-w) / tau_w : 1
                m_inf = 0.5 * (1 + tanh( (v-V_1)/V_2 )) : 1
                w_inf = 0.5 * (1 + tanh( (v-V_3)/V_4 )) : 1
                tau_w = ms / cosh( (v-V_3)/(2*V_4) ) : second
                V_3 : volt
                dI_noise/dt = -I_noise/tauxi + sigmaV*xi*tauxi**-0.5 :amp
                dy2_exc/dt = -y2_exc/Tau_epsc_2 + y_exc/Tau_epsc_1 : 1
                dy_exc/dt = -y_exc/Tau_epsc_1: 1
                dy2_inh/dt = -y2_inh/Tau_epsc_2 + y_inh/Tau_epsc_1 : 1
                dy_inh/dt = -y_inh/Tau_epsc_1: 1
                sigmaV : amp
                g_syn_exc : siemens
                g_syn_inh : siemens
            '''

            #############################Neuron group Defining##############################

            G_exc = NeuronGroup(group_size_exc * n_groups, eqs, threshold='v>V_thresh', method='heun',
                                refractory=3.3 * ms)
            G_exc.v = E_l  # initialize rest potential
            G_exc.w = 0  # initialize potasium channel

            # define b_w(namely, V_3) of each layer
            for z in range(n_groups):
                if z > 0:
                    if z % 2 == 0:  # L1,L3,...
                        G_exc.V_3[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['DIFF']['betaw'] * mV
                        G_exc.g_syn_exc[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['DIFF']['gsyn_exc'] * uS
                        G_exc.g_syn_inh[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['DIFF']['gsyn_inh'] * uS
                        G_exc.sigmaV[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['DIFF']['sigmaV']
                    if z % 2 == 1:  # L2,L4,...
                        G_exc.V_3[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INT']['betaw'] * mV
                        G_exc.g_syn_exc[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INT']['gsyn_exc'] * uS
                        G_exc.g_syn_inh[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INT']['gsyn_inh'] * uS
                        G_exc.sigmaV[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INT']['sigmaV']

                        # input layer
            G_exc.V_3[0 * group_size_exc:(0 + 1) * group_size_exc:1] = specs['INPUT']['betaw'] * mV
            G_exc.g_syn_exc[0 * group_size_exc:(0 + 1) * group_size_exc:1] = specs['INPUT']['gsyn_exc'] * uS
            G_exc.g_syn_inh[0 * group_size_exc:(0 + 1) * group_size_exc:1] = 0 * uS
            G_exc.sigmaV[0 * group_size_exc:(0 + 1) * group_size_exc:1] = specs['INPUT']['sigmaV']

            G_inh = NeuronGroup(group_size_inh * (n_groups - 1), eqs, threshold='v>V_thresh', method='heun',
                                refractory=3.3 * ms)
            G_inh.v = E_l  # initialize rest potential
            G_inh.w = 0  # initialize potasium channel
            G_inh.g_syn_inh = 0 * uS

            for z in range(n_groups - 1):
                if z % 2 == 0:  # L1.5,L3.5,...
                    G_inh.V_3[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INH']['betaw'] * mV
                    G_inh.g_syn_exc[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INH'][
                                                                                         'gsyn_from_odd'] * uS  # the coefficient of EPSC receiption
                    G_inh.sigmaV[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INH']['sigmaV']
                if z % 2 == 1:  # L2.5,L4.5,...
                    G_inh.V_3[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INH']['betaw'] * mV
                    G_inh.g_syn_exc[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INH'][
                                                                                         'gsyn_from_even'] * uS  # the coefficient of EPSC receiption
                    G_inh.sigmaV[z * group_size_exc:(z + 1) * group_size_exc:1] = specs['INH']['sigmaV']

                    #############################inter-group synapses##############################

            ### exc 2 exc ###
            S_e2e = Synapses(G_exc, G_exc, 'w_syn: 1', on_pre='y_exc_post += w_syn')
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

            ### exc 2 inh ###
            S_e2i = Synapses(G_exc, G_inh, 'w_syn: 1', on_pre='y_exc_post += w_syn')
            for z in range(n_groups - 1):
                if z > 0:
                    if z % 2 == 0:  # L1->L2, L3->L4, ... This should be sparser than the other case
                        S_e2i.connect(
                            condition='i>z*group_size_exc-1 and i<(z+1)*group_size_exc and j>z*group_size_inh-1 and j<(z+1)*group_size_inh',
                            p=specs['INH']['p'])
                    if z % 2 == 1:  # L2->L3, L4->L5, ...
                        S_e2i.connect(
                            condition='i>z*group_size_exc-1 and i<(z+1)*group_size_exc and j>z*group_size_inh-1 and j<(z+1)*group_size_inh',
                            p=specs['INH']['p'])

            S_e2i.connect(
                condition='i>z*group_size_exc-1 and i<(z+1)*group_size_exc and j>z*group_size_inh-1 and j<(z+1)*group_size_inh',
                p=specs['INH']['p'])

            S_e2i.delay = D  # fixed delay
            S_e2i.w_syn = 1  # ' 2 * rand()' # random synaptical weight with the mean of 1

            ### inh 2 exc ###
            S_i2e = Synapses(G_inh, G_exc, 'w_syn: 1', on_pre='y_inh_post += w_syn')
            for z in range(n_groups - 1):
                if z % 2 == 0:  # L1.5->L2, L3.5->L4, ... This should be sparser than the other case
                    S_i2e.connect(
                        condition='i>z*group_size_inh-1 and i<(z+1)*group_size_inh and j>(z+1)*group_size_exc-1 and j<(z+2)*group_size_exc',
                        p=specs['INH']['p'])
                if z % 2 == 1:  # L0.5->L1, L2.5->L3, ...
                    S_i2e.connect(
                        condition='i>z*group_size_inh-1 and i<(z+1)*group_size_inh and j>(z+1)*group_size_exc-1 and j<(z+2)*group_size_exc',
                        p=specs['INH']['p'])

            S_i2e.delay = D_inh  # fixed delay
            S_i2e.w_syn = 1  # ' 2 * rand()' # random synaptical weight with the mean of 1

            #############################Spike Generating##############################
            tmp1 = numpy.arange(group_size_exc)
            numpy.random.shuffle(tmp1)
            tmp2 = tmp1[:n_spikes]  # a trick to get n_spikes non-repeating random int

            Ginput_exc = SpikeGeneratorGroup(group_size_exc, tmp2, np.random.randn(n_spikes) * sigma + 100 * ms)

            Sinput_exc = Synapses(Ginput_exc, G_exc[:group_size_exc], on_pre='v_post += 1.0*(V_thresh-V_rest)')

            Sinput_exc.connect(j='i')
            Sinput_exc.delay = D

            Sp_exc = SpikeMonitor(G_exc)
            Sp_inh = SpikeMonitor(G_inh)

            #############################Run and plot##############################
            #             seed(4321)
            #             np.random.seed(4321)
            run(duration)

            #######################formatize data#########################

            tmp_t = Sp_exc.t / ms
            tmp_i = array(Sp_exc.i) + 1  # due to that matlab index start from 1
            spike_num = size(Sp_exc.t)
            T_sp_exc[0:spike_num:1, counter] = tmp_t.reshape([spike_num])
            I_sp_exc[0:spike_num:1, counter] = tmp_i.reshape([spike_num])

            tmp_t = Sp_inh.t / ms
            tmp_i = array(Sp_inh.i) + 1  # due to that matlab index start from 1
            spike_num = size(Sp_inh.t)
            T_sp_inh[0:spike_num:1, counter] = tmp_t.reshape([spike_num])
            I_sp_inh[0:spike_num:1, counter] = tmp_i.reshape([spike_num])

            counter += 1
            # print "progress: %d/%d" % (counter, np.size(Nspikes) * np.size(sigmas)), '\r',

    data_exc = {'Nspikes': np.float32(Nspikes),
                'sigmas': sigmas / ms,
                'groupsize': group_size_exc,
                'ngroups': n_groups,
                'spikei': I_sp_exc,
                'spiketime': T_sp_exc,
                'duration': duration / ms,
                'Kinh': 0,
                'Parameters': specs,
                'TauSynapse': [Tau_epsc_1, Tau_epsc_2],
                'DT': defaultclock.dt / ms}
    data_inh = {'Nspikes': np.float32(Nspikes),
                'sigmas': sigmas / ms,
                'groupsize': group_size_inh,
                'ngroups': n_groups - 1,
                'spikei': I_sp_inh,
                'spiketime': T_sp_inh,
                'duration': duration / ms,
                'Kinh': 0,
                'Parameters': specs,
                'TauSynapse': [Tau_epsc_1, Tau_epsc_2],
                'DT': defaultclock.dt / ms}
    return data_exc, data_inh


# # Heterogeneous FFN

group_size_exc = 4000
pbase = 9.0 / group_size_exc
convergence = 1.0

specs = {'nlayers': 9,
         'group_size_exc': group_size_exc,
         'group_size_inh': group_size_inh,
         'INT': {'betaw': 5.0, 'gsyn_exc': 345, 'gsyn_inh': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn_exc': 975, 'gsyn_inh': 975, 'sigmaV': '15.0*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn_exc': 975, 'gsyn_inh': 0, 'sigmaV': '38.0*uA', 'p':  pbase},
         'INH': {'betaw': betaw_inh, 'gsyn_from_odd': g_inh, 'gsyn_from_even': g_inh, 'sigmaV': '38.0*uA',
                 'p': pbase}}

sigma_list = np.linspace(2, 16, 8)
alpha_list = np.int32(np.linspace(group_size_exc/10, 8*group_size_exc/10, 8))

[data_exc, data_inh] = FFN_MP_run(sigma_list, alpha_list, specs)

sio.savemat(savePath + '/HetFFN_FFInh_exc.mat', data_exc)
sio.savemat(savePath + '/HetFFN_FFInh_inh.mat', data_inh)

# # DIF FFN


group_size_exc = 4000
pbase = 9.0 / group_size_exc
convergence = 1.0

specs = {'nlayers': 9,
         'group_size_exc': group_size_exc,
         'group_size_inh': group_size_inh,
         'INT': {'betaw': -19.0, 'gsyn_exc': 975, 'gsyn_inh': 975, 'sigmaV': '15.0*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn_exc': 975, 'gsyn_inh': 975, 'sigmaV': '15.0*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn_exc': 975, 'gsyn_inh': 0, 'sigmaV': '38.0*uA', 'p':  pbase},
         'INH': {'betaw': betaw_inh, 'gsyn_from_odd': g_inh, 'gsyn_from_even': g_inh, 'sigmaV': '38.0*uA',
                 'p': pbase}}

sigma_list = np.linspace(2, 16, 8)
alpha_list = np.int32(np.linspace(group_size_exc/10, 8*group_size_exc/10, 8))

[data_exc, data_inh] = FFN_MP_run(sigma_list, alpha_list, specs)
sio.savemat(savePath + '/DIFFFN_FFInh_exc.mat', data_exc)
sio.savemat(savePath + '/DIFFFN_FFInh_inh.mat', data_inh)

# # INT FFN


group_size_exc = 4000
pbase = 9.0 / group_size_exc
convergence = 1.0

specs = {'nlayers': 9,
         'group_size_exc': group_size_exc,
         'group_size_inh': group_size_inh,
         'INT': {'betaw': 5.0, 'gsyn_exc': 345, 'gsyn_inh': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0, 'gsyn_exc': 345, 'gsyn_inh': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn_exc': 975, 'gsyn_inh': 0, 'sigmaV': '38.0*uA', 'p':  pbase},
         'INH': {'betaw': betaw_inh, 'gsyn_from_odd': g_inh, 'gsyn_from_even': g_inh, 'sigmaV': '38.0*uA',
                 'p': pbase}}

sigma_list = np.linspace(2, 16, 8)
alpha_list = np.int32(np.linspace(group_size_exc/10, 8*group_size_exc/10, 8))

[data_exc, data_inh] = FFN_MP_run(sigma_list, alpha_list, specs)
sio.savemat(savePath + '/INTFFN_FFInh_exc.mat', data_exc)
sio.savemat(savePath + '/INTFFN_FFInh_inh.mat', data_inh)



