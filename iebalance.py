#!/usr/bin/env python

import brian as b

tau = 20 * b.msecond  # membrane time constant
V_rest = -60 * b.mvolt  # resting membrane potential
Theta = -50 * b.mvolt  # spiking threshold
tau_ref = 5 * b.msecond  # refractory period
g_leak = 10 * b.nsiemens  # leak conductance

V_E = 0 * b.mvolt  # excitatory reversal potential
V_I = -80 * b.mvolt  # inhibitory reversal potential
tau_E = 5 * b.msecond  # excitatory synaptic time constant
tau_I = 10 * b.msecond  # inhibitory synaptic time constant

eqs = b.Equations('''
    tau * dV/dt = \
        (V_rest - V) + (g_E * (V_E - V) + g_I * (V_I - V) + I_b) / g_leak
    tau_E dg_E/dt = -g_E
    tau_I dg_I/dt = -g_I
    ''')


if __name__ == '__main__':
    pass
