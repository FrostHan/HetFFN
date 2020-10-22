# coding: utf-8

from brian2 import *
import numpy as np
import scipy.io as sio
import os
import time

seed(4321)
np.random.seed(4321)

# determine the save path
savePath = './data/deepFFN'

if os.path.exists(savePath):
    warnings.warn('{} exists (possibly so do data).'.format(savePath))
else:
    os.makedirs(savePath)


# Simulate the FFN with given beta_w and synaptic conductance

def FFN_MP_run(sigma_list, alpha_list, specs):

    # simulation dt
    defaultclock.dt = 0.01 * ms

    # Simulation parameters
    duration = 180 * ms
    sigmas = sigma_list * ms  # sigma of input stimuli
    Nspikes = alpha_list

    # Network macro parameter

    n_groups = specs['nlayers']
    group_size_exc = specs['group_size']

    V_thresh = -10 * mV
    V_rest = -80 * mV

    D = 0 * ms  # delay

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
    I_sp = np.zeros([800000, size(sigmas) * size(Nspikes)])
    T_sp = np.zeros([800000, size(sigmas) * size(Nspikes)])

    tauxi = 1.0 * ms  # noise time scale

    counter = 0
    max_spike_num = 0

    for sigma in sigmas:
        for n_spikes in Nspikes:

            n_spikes = int(n_spikes)

            start_scope()

            # M-L model
            eqs_exc = '''
                dv/dt = (-g_l*(v-E_l) - g_Na*m_inf*(v-E_Na) - g_K*w*(v-E_K) + g_syn*y2*(E_exc-v)/cm**2 + I_noise/cm**2) / C : volt
                dw/dt = phi*(w_inf-w) / tau_w : 1
                m_inf = 0.5 * (1 + tanh( (v-V_1)/V_2 )) : 1
                w_inf = 0.5 * (1 + tanh( (v-V_3)/V_4 )) : 1
                tau_w = ms / cosh( (v-V_3)/(2*V_4) ) : second
                V_3 : volt
                dI_noise/dt = -I_noise/tauxi + sigmaV*xi*tauxi**-0.5 :amp
                dy2/dt = -y2/Tau_epsc_2 + y/Tau_epsc_1 : 1
                dy/dt = -y/Tau_epsc_1: 1
                sigmaV : amp
                g_syn : siemens
            '''

            #############################Neuron group Defining##############################

            G_exc = NeuronGroup(group_size_exc * n_groups, eqs_exc, threshold='v>V_thresh', method='heun',
                                refractory=3.3 * ms)
            G_exc.v = E_l  # initialize rest potential
            G_exc.w = 0  # initialize synaptic weight
            G_exc.g_syn = 0 * uS  # initialize synaptic conductance

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

            #############################Spike Generating##############################
            tmp1 = numpy.arange(group_size_exc)
            numpy.random.shuffle(tmp1)
            tmp2 = tmp1[:n_spikes]  # a trick to get n_spikes non-repeating random int

            Ginput_exc = SpikeGeneratorGroup(group_size_exc, tmp2, np.random.randn(n_spikes) * sigma + 100 * ms)

            Sinput_exc = Synapses(Ginput_exc, G_exc[:group_size_exc], on_pre='v_post += 1.0*(V_thresh-V_rest)')

            Sinput_exc.connect(j='i')
            Sinput_exc.delay = D

            Sp_exc = SpikeMonitor(G_exc)

            #############################Run and plot##############################

            run(duration)

            #######################formatize data#########################

            tmp_t = Sp_exc.t / ms
            tmp_i = array(Sp_exc.i) + 1  # due to that matlab index start from 1
            spike_num = size(Sp_exc.t)
            T_sp[0:spike_num:1, counter] = tmp_t.reshape([spike_num])
            I_sp[0:spike_num:1, counter] = tmp_i.reshape([spike_num])

            if spike_num > max_spike_num:
                max_spike_num = spike_num
            counter += 1
            print(counter)

    T_sp = T_sp[0:max_spike_num]
    I_sp = I_sp[0:max_spike_num]

    data = {'Nspikes': np.float32(Nspikes),
            'sigmas': sigmas / ms,
            'groupsize': group_size_exc,
            'ngroups': n_groups,
            'spikei': I_sp,
            'spiketime': T_sp,
            'duration': duration / ms,
            'Kinh': 0,
            'Parameters': specs,
            'TauSynapse': [Tau_epsc_1, Tau_epsc_2],
            'DT': defaultclock.dt / ms}
    return (data)


# ---------------------- For Fig. 3 ----------------------------

# # Heterogeneous FFN

group_size = 1000
pbase = 9.0/group_size
convergence = 1

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': 5.0,   'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',   'p': pbase}
        }

data = FFN_MP_run([5], [400], specs)

sio.savemat(savePath+'/HetFFN_5ms_400spks.mat',data)



group_size = 1000
pbase = 9.0/group_size
convergence = 1/1

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': 5.0,   'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',   'p': pbase}
        }

data = FFN_MP_run([14], [1000], specs)

sio.savemat(savePath+'/HetFFN_14ms_1000spks.mat',data)


group_size = 1000
pbase = 9.0/group_size
convergence = 1/1

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': 5.0,   'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',   'p': pbase}
        }

data = FFN_MP_run([15], [1000], specs)

sio.savemat(savePath+'/HetFFN_15ms_1000spks.mat',data)


# Assign beta_w
convergence = 1/1
specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': 5.0,   'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',   'p': pbase}
        }

data = FFN_MP_run([1], [200], specs)

sio.savemat(savePath+'/HetFFN_1ms_200spks.mat',data)


# # DIF FFN


specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': pbase}
        }

data = FFN_MP_run([5], [400], specs)

sio.savemat(savePath+'/DIFFFN_5ms_400spks.mat',data)


specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': pbase}
        }

data = FFN_MP_run([14], [1000], specs)

sio.savemat(savePath+'/DIFFFN_14ms_1000spks.mat',data)


specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': 40.0/group_size}
        }

data = FFN_MP_run([15], [500], specs)

sio.savemat(savePath+'/DIFFFN_15ms_500spks.mat',data)


convergence = 1

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': pbase}
        }

data = FFN_MP_run([1], [200], specs)

sio.savemat(savePath+'/DIFFFN_1ms_200spks.mat',data)


# # INT FFN


specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA',  'p': pbase},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': pbase}
        }

data = FFN_MP_run([5], [400], specs)

sio.savemat(savePath+'/INTFFN_5ms_400spks.mat',data)



specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA',  'p': pbase},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': pbase}
        }

data = FFN_MP_run([15], [1000], specs)
sio.savemat(savePath+'/INTFFN_15ms_1000spks.mat',data)



specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA',  'p': pbase},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': 40.0/group_size}
        }

data = FFN_MP_run([15], [500], specs)
sio.savemat(savePath+'/INTFFN_15ms_500spks.mat',data)


data = FFN_MP_run([1], [60], specs)
sio.savemat(savePath+'/INTFFN_1ms_60spks.mat',data)


specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA',  'p': pbase},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': pbase}
        }

data = FFN_MP_run([1], [200], specs)


sio.savemat(savePath+'/INTFFN_1ms_200spks.mat',data)



# --------------- For Fig. 4 --------------------

# # Heterogeneous FFN

group_size = 10000
pbase = 9.0/group_size
convergence = 9.0/9.0

specs = {'nlayers': 13,
         'group_size': group_size,
         'INT':  {'betaw': 5.0,   'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',   'p': pbase}
        }

sigma_list = np.array([1.0])
alpha_list = np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]) * group_size

data = FFN_MP_run(sigma_list, alpha_list, specs)
sio.savemat(savePath + '/HetFFN_delay.mat',data)


# # DIF FFN

del data
specs = {'nlayers': 13,
         'group_size': group_size,
         'INT':  {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': pbase}
        }

sigma_list = np.array([1.0])
alpha_list = np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]) * group_size

data = FFN_MP_run(sigma_list, alpha_list, specs)
sio.savemat(savePath + '/DIFFFN_delay.mat',data)


# # INT FFN

del data
specs = {'nlayers': 13,
         'group_size': group_size,
         'INT':  {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',  'p': pbase}
        }

sigma_list = np.array([1.0])
alpha_list = np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]) * group_size

data = FFN_MP_run(sigma_list, alpha_list, specs)
sio.savemat(savePath + '/INTFFN_delay.mat',data)



# ---------------------- For supplementary figure A5 ----------------------------


# # Homo InterMed FFN

group_size = 1000
pbase = 9.0/group_size
convergence = 1

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': -10.0,   'gsyn': 415, 'sigmaV': '38.0*uA', 'p': pbase},
         'DIFF': {'betaw': -10.0, 'gsyn': 415, 'sigmaV': '38.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',   'p': pbase}
        }

data = FFN_MP_run([2, 5], [400, 900], specs)
sio.savemat(savePath+'/MedHomFFN_-10.mat',data)


group_size = 1000
pbase = 9.0/group_size
convergence = 1

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': -12.0,   'gsyn': 435, 'sigmaV': '38.0*uA', 'p': pbase},
         'DIFF': {'betaw': -12.0, 'gsyn': 435, 'sigmaV': '38.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',   'p': pbase}
        }

data = FFN_MP_run([2, 5], [400, 900], specs)
sio.savemat(savePath+'/MedHomFFN_-12.mat',data)



group_size = 1000
pbase = 9.0/group_size
convergence = 1

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': -7.0,   'gsyn': 375, 'sigmaV': '38.0*uA', 'p': pbase},
         'DIFF': {'betaw': -7.0, 'gsyn': 375, 'sigmaV': '38.0*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA',   'p': pbase}
        }

data = FFN_MP_run([2, 5], [400, 900], specs)
sio.savemat(savePath+'/MedHomFFN_-7.mat',data)



# # Reversed Heterogeneous FFN

group_size = 1000
pbase = 9.0/group_size
convergence = 1

specs = {'nlayers': 7,
         'group_size': group_size,
         'INT':  {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0,   'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA',   'p': pbase*convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA', 'p': pbase}
        }

data = FFN_MP_run([2, 5], [500, 900], specs)
sio.savemat(savePath+'/RevHetFFN.mat',data)


# ------------------------- For supplementary figure A2 -------------------
# # Heterogeneous FFN

group_size = 5000
pbase = 9.0 / group_size
convergence = 9.0 / 9.0
specs = {'nlayers': 9,
         'group_size': group_size,
         'INT': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA', 'p': pbase}
         }
sigma_list = np.linspace(2, 16, 8)
alpha_list = np.int32(np.linspace(group_size / 10, 8 * group_size / 10, 8))
data = FFN_MP_run(sigma_list, alpha_list, specs)
sio.savemat(savePath + '/HetFFN_MP.mat', data)


# # DIF FFN
specs = {'nlayers': 9,
         'group_size': group_size,
         'INT': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA', 'p': pbase},
         'DIFF': {'betaw': -19.0, 'gsyn': 975, 'sigmaV': '15.0*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA', 'p': pbase}
         }
sigma_list = np.linspace(2, 16, 8)
alpha_list = np.int32(np.linspace(group_size / 10, 8 * group_size / 10, 8))
data = FFN_MP_run(sigma_list, alpha_list, specs)
sio.savemat(savePath + '/DIFFFN_MP.mat', data)



# # INT FFN
specs = {'nlayers': 9,
         'group_size': group_size,
         'INT': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase},
         'DIFF': {'betaw': 5.0, 'gsyn': 345, 'sigmaV': '(38.0+15*rand())*uA', 'p': pbase * convergence},
         'INPUT': {'betaw': -23.0, 'gsyn': 975, 'sigmaV': '38.0*uA', 'p': pbase}
         }
sigma_list = np.linspace(2, 16, 8)
alpha_list = np.int32(np.linspace(group_size / 10, 8 * group_size / 10, 8))
data = FFN_MP_run(sigma_list, alpha_list, specs)
sio.savemat(savePath + '/INTFFN_MP.mat', data)
