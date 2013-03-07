#!/usr/bin/env python

import brian as b
import numpy as np
import numpy.random as rnd

tau = 20 * b.msecond  # membrane time constant
V_rest = -60 * b.mvolt  # resting membrane potential
Theta = -50 * b.mvolt  # spiking threshold
tau_ref = 5 * b.msecond  # refractory period
g_leak = 10 * b.nsiemens  # leak conductance

V_E = 0 * b.mvolt  # excitatory reversal potential
V_I = -80 * b.mvolt  # inhibitory reversal potential
tau_E = 5 * b.msecond  # excitatory synaptic time constant
tau_I = 10 * b.msecond  # inhibitory synaptic time constant

I_b = 0 * b.amp

eqs = b.Equations('''
    I_E = g_E * (V_E - V) : amp
    I_I = g_I * (V_I - V) : amp
    dV/dt = ((V_rest - V) + (I_E + I_I + I_b) / g_leak) / tau : volt
    dg_E/dt = -g_E / tau_E : siemens
    dg_I/dt = -g_I / tau_I : siemens
    ''')

neuron = b.NeuronGroup(1, model=eqs, reset=V_rest, threshold=Theta)

M_E = b.StateMonitor(neuron, 'I_E', record=True)
M_I = b.StateMonitor(neuron, 'I_I', record=True)


def tuning_function(subgroup_indices, peak):
    return 0.3 + rnd.rand(*subgroup_indices.shape) / 10.0 + \
        1.1 / (1.0 + np.absolute(subgroup_indices + 1 - peak)) ** 4


if __name__ == '__main__':
    import pickle
    import sys

    with open(sys.argv[1], 'r') as f:
        spiketimes = pickle.load(f)
    N = max(zip(*spiketimes)[0]) + 1
    G = b.SpikeGeneratorGroup(N, spiketimes, sort=False, period=500 * b.msecond)

    fraction_inhibitory = 0.2
    excitatory = G.subgroup(int((1.0 - fraction_inhibitory) * N))
    inhibitory = G.subgroup(int(fraction_inhibitory * N))

    num_subgroups = 8
    excitatory_subgroup_size = len(excitatory) // num_subgroups
    inhibitory_subgroup_size = len(inhibitory) // num_subgroups
    excitatory_subgroups = [excitatory.subgroup(excitatory_subgroup_size)
                            for i in xrange(num_subgroups)]
    inhibitory_subgroups = [inhibitory.subgroup(inhibitory_subgroup_size)
                            for i in xrange(num_subgroups)]

    excitatory_connections = b.Connection(excitatory, neuron, 'g_E')

    subgroup_indices = np.concatenate([
        np.repeat(i, excitatory_subgroup_size) for i in xrange(num_subgroups)])
    excitatory_connections.connect(
        excitatory, neuron, 140 * b.psiemens *
        np.atleast_2d(tuning_function(subgroup_indices, 5)).T)

    g_I_bar = 350 * b.psiemens
    inhibitory_connections = b.Connection(inhibitory, neuron, 'g_I')
    inhibitory_connections[:, :] = 0.1 * g_I_bar

    tau_stdp = 20 * b.msecond
    eta = 1e-4
    rho = 5 / b.second
    alpha = 2 * rho * tau_stdp
    eqs_stdp = '''
        dx_pre/dt = -x_pre / tau_stdp : 1
        dx_post/dt = -x_post / tau_stdp : 1
        '''
    stdp = b.STDP(
        inhibitory_connections, eqs=eqs_stdp,
        pre='x_pre += 1; w += g_I_bar * eta * (x_pre - alpha)',
        post='x_post += 1; w += g_I_bar * eta * x_post')

    net = b.Network(
        neuron, M_E, M_I, G, excitatory_connections, inhibitory_connections,
        stdp)
