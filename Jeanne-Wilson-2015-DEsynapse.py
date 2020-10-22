# coding: utf-8

# This program simulates the signal processing in drosophila olfactry pathway.

from docopt import docopt

from brian2 import *
from utils import *
import numpy as np
import scipy.io as sio
import os, time, warnings

seed(42)
np.random.seed(42)

# Determine where the .mat data saved
savePath = './data/ALFFN'
if os.path.exists(savePath):
    warnings.warn('{} exists (possibly so do data).'.format(savePath))
else:
    os.makedirs(savePath)

# functions defining and main program:
def droso_ffn_run(I_inj, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN):
    #     Arguments:
    #     ORNV_3: beta_w of ORN layer
    #     PNV_3: beta_w of PN layer
    #     LHNV_3: beta_w of LHN layer
    #     g_exc_12: synaptic conductance between ORN layer and PN layer
    #     g_exc_23: synaptic conductance between PN layer and LHN layer
    #     sigmaV_ORN: noise level of ORN neurons
    #     sigmaV_PN: noise level of PN neurons
    #     sigmaV_LHN: noise level of LHN neurons

    defaultclock.dt = 0.01 * ms
    start_scope()

    # Simulation parameters
    duration = 400 * ms

    # Network macro parameter
    N_ORN = 4000
    N_PN = 900
    N_LHN = 900
    p_connect_12 = 0.01
    p_connect_23 = 0.01

    n_spikes = N_ORN
    sigma = 0.0 * ms
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
    V_thresh = -10 * mV

    tauxi = 1.0 * ms  # noise time scale

    # paramter of input stimulus (double-exponential)
    Tau_inj1 = 15 * ms
    Tau_inj2 = 50 * ms

    #############################model eqs##############################

    # conductance-based input
    eqs_ORN = '''
    dv/dt = (-g_l*(v-E_l) - g_Na*m_inf*(v-E_Na) - g_K*w*(v-E_K) + I_syn/cm**2 + I_noise/cm**2 + Ib_ORN/cm**2) / C : volt
    dw/dt = phi*(w_inf-w) / tau_w : 1
    m_inf = 0.5 * (1 + tanh( (v-V_1)/V_2 )) : 1
    w_inf = 0.5 * (1 + tanh( (v-V_3)/V_4 )) : 1
    tau_w = ms / cosh( (v-V_3)/(2*V_4) ) : second
    V_3 : volt
    dI_syn/dt = -I_syn/Tau_inj2 + y/Tau_inj1 :amp
    dI_noise/dt = -I_noise/tauxi + sigmaV_ORN*xi*tauxi**-0.5 :amp
    dy/dt = - y/Tau_inj1 :amp
    sigmaV_ORN :amp
    '''
    # y=H(t-t0)*I_inj*Exp[-(t-t0)/Tau_inj1]
    # I_syn=I_inj*{Exp[-(t-t0)/Tau_inj2]-Exp[-(t-t0)/Tau_inj1]}

    eqs_PN = '''
    dv/dt = (-g_l*(v-E_l) - g_Na*m_inf*(v-E_Na) - g_K*w*(v-E_K) + g_syn*(E_exc-v)/cm**2 + I_noise/cm**2 + Ib_PN/cm**2) / C : volt
    dw/dt = phi*(w_inf-w) / tau_w : 1
    m_inf = 0.5 * (1 + tanh( (v-V_1)/V_2 )) : 1
    w_inf = 0.5 * (1 + tanh( (v-V_3)/V_4 )) : 1
    tau_w = ms / cosh( (v-V_3)/(2*V_4) ) : second
    V_3 : volt
    dI_noise/dt = -I_noise/tauxi + sigmaV_PN*xi*tauxi**-0.5 :amp
    dg_syn/dt = -g_syn/Tau_epsc_2 + y/Tau_epsc_1 : siemens
    dy/dt = -y/Tau_epsc_1: siemens
    sigmaV_PN : amp
    '''

    eqs_LHN = '''
    dv/dt = (-g_l*(v-E_l) - g_Na*m_inf*(v-E_Na) - g_K*w*(v-E_K) + g_syn*(E_exc-v)/cm**2 + I_noise/cm**2 + Ib_LHN/cm**2) / C : volt
    dw/dt = phi*(w_inf-w) / tau_w : 1
    m_inf = 0.5 * (1 + tanh( (v-V_1)/V_2 )) : 1
    w_inf = 0.5 * (1 + tanh( (v-V_3)/V_4 )) : 1
    tau_w = ms / cosh( (v-V_3)/(2*V_4) ) : second
    V_3 : volt
    dI_noise/dt = -I_noise/tauxi + sigmaV_LHN*xi*tauxi**-0.5 :amp
    dg_syn/dt = -g_syn/Tau_epsc_2 + y/Tau_epsc_1 : siemens
    dy/dt = -y/Tau_epsc_1: siemens
    sigmaV_LHN: amp
    '''

    # realize Alpha function by introducing variable x and y
    # I_syn: Synaptic current input (PSC)
    # y: g_syn*(v_pre-v_post)*exp(-(t-t_spike)/Tau)

    #############################Neuron group Defining##############################
    ORN = NeuronGroup(N_ORN, eqs_ORN, threshold='v>V_thresh', method='heun', refractory=3.3 * ms)
    PN = NeuronGroup(N_PN, eqs_PN, threshold='v>V_thresh', method='heun', refractory=3.3 * ms)
    LHN = NeuronGroup(N_LHN, eqs_LHN, threshold='v>V_thresh', method='heun', refractory=3.3 * ms)

    ORN.v = E_l
    PN.v = E_l
    LHN.v = E_l

    ORN.w = 0
    PN.w = 0
    LHN.w = 0

    ORN.I_syn = 0 * pA
    PN.g_syn = 0 * nS
    LHN.g_syn = 0 * nS

    ORN.V_3 = ORNV_3
    PN.V_3 = PNV_3
    LHN.V_3 = LHNV_3
    ORN.sigmaV_ORN = sigmaV_ORN
    PN.sigmaV_PN = sigmaV_PN
    LHN.sigmaV_LHN = sigmaV_LHN
    #############################Spike Generating##############################

    Ginput1 = SpikeGeneratorGroup(n_spikes, np.arange(n_spikes), np.random.randn(n_spikes) * sigma + 200 * ms)
    Sinput1 = Synapses(Ginput1, ORN, 'w_syn :1',
                       on_pre='''
                      y_post += I_inj
                      ''')

    Ginput2 = SpikeGeneratorGroup(n_spikes, np.arange(n_spikes), np.random.randn(n_spikes) * sigma + 300 * ms)
    Sinput2 = Synapses(Ginput2, ORN, 'w_syn :1',
                       on_pre='''
                      I_syn_post += 0 *uA
                      ''')

    Sinput1.connect(j='i')
    Sinput1.delay = 0.5 * ms

    Sinput2.connect(j='i')
    Sinput2.delay = 0.5 * ms

    #############################Recoding##############################


    Sp_ORN = SpikeMonitor(ORN, record=range(30))
    Sp_PN = SpikeMonitor(PN, record=range(30))
    Sp_LHN = SpikeMonitor(LHN, record=range(30))

    St_ORN = StateMonitor(ORN, 'v', record=range(100))
    St_PN = StateMonitor(PN, 'v', record=range(100))
    St_LHN = StateMonitor(LHN, 'v', record=range(100))

    St_I = StateMonitor(ORN, 'I_syn', record=range(30))

    #############################inter-group synapses##############################

    S1 = Synapses(ORN, PN, 'w_syn :1', on_pre='y_post += w_syn * g_exc_12 ')
    S1.connect(p=p_connect_12)
    # S1.connect(condition='j<=i/40.0<j+1')
    S1.w_syn = 1.0  # ' 2 * rand()'

    S2 = Synapses(PN, LHN, 'w_syn :1', on_pre='y_post += w_syn * g_exc_23 ')
    S2.connect(p=p_connect_23)
    # S2.connect(condition='j<=i/9.0<j+1')
    S2.w_syn = 1.0  # ' 2 * rand()'
    #############################Run and plot##############################


    run(duration)
    data = {'betaw': [ORNV_3, PNV_3, LHNV_3] / mV,
            'sigmaV_ORN': sigmaV_ORN,
            'sigmaV_LHN': sigmaV_LHN,
            'sigmaV_PN': sigmaV_PN,
            'g_exc': [g_exc_12, g_exc_23] / uS,
            'NPN': N_PN,
            'NORN': N_ORN,
            'NLHN': N_LHN,
            'VORN': St_ORN.v / mV,
            'VORNT': St_ORN.t / ms,
            'VPN': St_PN.v / mV,
            'VPNT': St_PN.t / ms,
            'VLHN': St_LHN.v / mV,
            'VLHNT': St_LHN.t / ms,
            'TORN': Sp_ORN.t / ms,
            'IORN': array(Sp_ORN.i),
            'TPN': Sp_PN.t / ms,
            'IPN': array(Sp_PN.i),
            'TLHN': Sp_LHN.t / ms,
            'ILHN': array(Sp_LHN.i),
            'duration': duration / ms,
            'DT': defaultclock.dt / ms,
            'source_index_ORN': np.array(S1.i),
            'target_index_PN': np.array(S1.j),
            'source_index_PN': np.array(S2.i),
            'target_index_LHN': np.array(S2.j)}

    return data


# Set additional parameters
sigmaV_INT = '(38+15*rand())*uA'
sigmaV_DIF = '15*uA'
Ib_INT = 0 * uA
Ib_DIF = 0 * uA

## Jeanne-Wilson-2015-Simulation-Het-Layers

g_exc_12 = 345 * uS
g_exc_23 = 975 * uS
sigmaV_ORN = '38.0 *uA'  # noise level
sigmaV_PN = sigmaV_INT  # noise level
sigmaV_LHN = sigmaV_DIF  # noise level
Ib_ORN = 0 * uA
Ib_PN = Ib_INT
Ib_LHN = Ib_DIF

ORNV_3 = -23 * mV  # class 3
PNV_3 = 5 * mV  # class 1
LHNV_3 = -19 * mV  # class 3
I_inj = 45 * uA

data = droso_ffn_run(I_inj, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN)
# Save data
sio.savemat(savePath + '/DrosoDataHet.mat', data)

data = droso_ffn_run(0 * uA, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN) # No stimuli
# Save data
sio.savemat(savePath + '/DrosoDataHet_NoStim.mat', data)


## Jeanne-Wilson-2015-Simulation-ALLDIF-Layers

g_exc_12 = 1170 * uS
g_exc_23 = 715 * uS
sigmaV_ORN = '38.0 *uA'  # noise level
sigmaV_PN = sigmaV_DIF  # noise level
sigmaV_LHN = sigmaV_DIF  # noise level
Ib_ORN = 0 * uA
Ib_PN = Ib_DIF
Ib_LHN = Ib_DIF

ORNV_3 = -23 * mV  # class 3
PNV_3 = -19 * mV  # class 1
LHNV_3 = -19 * mV  # class 3
I_inj = 45 * uA

data = droso_ffn_run(I_inj, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN)
# Save data
sio.savemat(savePath + '/DrosoDataDIF.mat', data)

data = droso_ffn_run(0 * uA, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN) # No stimuli
# Save data
sio.savemat(savePath + '/DrosoDataDIF_NoStim.mat', data)


## Jeanne-Wilson-2015-Simulation-ALLINT-Layers

g_exc_12 = 345 * uS
g_exc_23 = 285 * uS

sigmaV_ORN = '38.0 *uA'  # noise level
sigmaV_PN = sigmaV_INT  # noise level
sigmaV_LHN = sigmaV_INT  # noise level
Ib_ORN = 0 * uA
Ib_PN = Ib_INT
Ib_LHN = Ib_INT

ORNV_3 = -23 * mV  # class 3
PNV_3 = 5 * mV  # class 1
LHNV_3 = 5 * mV  # class 1
I_inj = 45 * uA

data = droso_ffn_run(I_inj, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN)
# Save data
sio.savemat(savePath + '/DrosoDataINT.mat', data)

data = droso_ffn_run(0 * uA, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN) # No stimuli
# Save data
sio.savemat(savePath + '/DrosoDataINT_NoStim.mat', data)


# Jeanne-Wilson-2015-Simulation-Reversed-Het-Layers

g_exc_12 = 975 * uS
g_exc_23 = 345 * uS

sigmaV_ORN = '38.0 *uA'  # noise level
sigmaV_PN = sigmaV_DIF  # noise level
sigmaV_LHN = sigmaV_INT  # noise level
Ib_ORN = 0 * uA
Ib_PN = Ib_INT
Ib_LHN = Ib_DIF

ORNV_3 = -23 * mV  # class 3
PNV_3 = -19 * mV  # class 1
LHNV_3 = 5 * mV  # class 3
I_inj = 45 * uA

data = droso_ffn_run(I_inj, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN)
# Save data
sio.savemat(savePath + '/DrosoDataRevHet.mat', data)

data = droso_ffn_run(0 * uA, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN)
# Save data
sio.savemat(savePath + '/DrosoDataRevHet_NoStim.mat', data)


## Jeanne-Wilson-2015-Simulation-Intermediate-betaw-Layers

sigmaV_MID = '38.0 *uA'

g_exc_12 = 375 * uS
g_exc_23 = 375 * uS
sigmaV_ORN = '38.0 *uA'  # noise level
sigmaV_PN = sigmaV_MID  # noise level
sigmaV_LHN = sigmaV_MID  # noise level

Ib_ORN = 0 * uA
Ib_PN = Ib_DIF
Ib_LHN = Ib_DIF

ORNV_3 = -23 * mV  # class 3
PNV_3 = -10 * mV  # class 1
LHNV_3 = -10 * mV  # class 3
I_inj = 45 * uA

data = droso_ffn_run(I_inj, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN)
# Save data
sio.savemat(savePath + '/DrosoDataMed.mat', data)

data = droso_ffn_run(0 * uA, ORNV_3, PNV_3, LHNV_3, g_exc_12, g_exc_23, sigmaV_ORN, sigmaV_PN, sigmaV_LHN)
# Save data
sio.savemat(savePath + '/DrosoDataMed_NoStim.mat', data)
