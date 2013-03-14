#!/usr/bin/env python

from config import Configurable, quantity, quantity_list
import brian as b
import inputs
import numpy as np
import numpy.random as rnd


class ModelNeuronGroupBuilder(Configurable):
    def __init__(self, config):
        Configurable.__init__(self, config)
        self._add_config_value('tau', quantity)  # membrane time constant
        self._add_config_value('V_rest', quantity)  # resting membrane potential
        self._add_config_value('Theta', quantity)  # spiking threshold
        self._add_config_value('tau_ref', quantity)  # refractory period
        self._add_config_value('g_leak', quantity)  # leak conductance
        self._add_config_value('refractory_period', quantity)

        self._add_config_value('V_E', quantity)  # excitatory reversal potential
        self._add_config_value('V_I', quantity)  # inhibitory reversal potential
        self._add_config_value('tau_E', quantity)  # excitat. syn. time constant
        self._add_config_value('tau_I', quantity)  # inhibit. syn. time constant
        self._add_config_value('I_b', quantity)  # additional current

        self.eqs = b.Equations('''
            I_E = g_E * (self.V_E - V) : amp
            I_I = g_I * (self.V_I - V) : amp
            dV/dt = ((self.V_rest - V) + (I_E + I_I + self.I_b) / self.g_leak) / \
                self.tau : volt
            dg_E/dt = -g_E / self.tau_E : siemens
            dg_I/dt = -g_I / self.tau_I : siemens
            ''')

    def build(self, N=1):
        return b.NeuronGroup(
            N, model=self.eqs, reset=self.V_rest, threshold=self.Theta,
            refractory=self.refractory_period)


class ModelGroups(object):
    def __init__(self, indexing, in_group):
        for key in indexing.iterkeys():
            assert np.all(1 == np.diff(
                [idx for group in indexing[key] for idx in group]))
            assert np.all(0 == np.diff([len(group) in group in indexing[key]]))

        self.excitatory = in_group.subgroup(
            sum(len(group) for group in indexing['excitatory']))
        self.inhibitory = in_group.subgroup(
            sum(len(group) for group in indexing['inhibitory']))

        self.ex_group_membership = np.hstack(
            np.repeat(i, len(group))
            for i, group in enumerate(indexing['excitatory']))
        self.inh_group_membership = np.hstack(
            np.repeat(i, len(group))
            for i, group in enumerate(indexing['excitatory']))


class ModelConnectionBuilder(Configurable):
    def __init__(self, config):
        Configurable.__init__(self, config)
        self._add_config_value('g_E_bar', quantity)
        self._add_config_value('g_I_bar', quantity)

    def build(self, groups, out_neuron):
        excitatory_connections = b.Connection(
            groups.excitatory, out_neuron, 'g_E')
        excitatory_connections.connect(
            groups.excitatory, neuron, self.g_E_bar *
            np.atleast_2d(tuning_function(groups.ex_group_membership, 5)).T)

        inhibitory_connections = b.Connection(
            groups.inhibitory, out_neuron, 'g_I', structure='dense')
        inhibitory_connections[:, :] = 0.1 * self.g_I_bar
        return excitatory_connections, inhibitory_connections


class ModelSTDPBuilder(Configurable):
    def __init__(self, config):
        Configurable.__init__(self, config)
        self._add_config_value('tau_stdp', quantity)
        self._add_config_value('eta', float)
        self._add_config_value('rho', quantity)
        self.alpha = 2 * self.rho * self.tau_stdp
        self.eqs = '''
            dx_pre/dt = -x_pre / self.tau_stdp : 1
            dx_post/dt = -x_post / self.tau_stdp : 1
            '''

    def build(self, connections, g_bar):
        eta = self.eta
        alpha = self.alpha
        assert eta, alpha  # suppress unused warnings
        return b.STDP(
            connections, eqs=self.eqs,
            pre='x_pre += 1; w += g_bar * eta * (x_pre - alpha)',
            post='x_post += 1; w += g_bar * eta * x_post')


def tuning_function(subgroup_indices, peak):
    return 0.3 + rnd.rand(*subgroup_indices.shape) / 10.0 + \
        1.1 / (1.0 + (subgroup_indices + 1 - peak) ** 4)


if __name__ == '__main__':
    import argparse
    import json
    import tables

    parser = argparse.ArgumentParser(
        description="Run the Vogels et al. 2011 single cell model.")
    parser.add_argument(
        '-c', '--config', type=str, nargs=1, required=True,
        help="Path to the configuration file.")
    parser.add_argument(
        'output', nargs=1, type=str,
        help="Path ot the HDF5 output file.")
    args = parser.parse_args()

    with open(args.config[0], 'r') as f:
        config = json.load(f)

    neuron = ModelNeuronGroupBuilder(config['model']).build()

    generator = inputs.GroupedSpikeTimesGenerator(config['inputs'])
    N = config['inputs']['num_trains']
    G = b.SpikeGeneratorGroup(N, inputs.swap_tuple_values(generator))

    indexing_scheme = generator.get_indexing_scheme()
    connection_builder = ModelConnectionBuilder(config['model'])
    groups = ModelGroups(indexing_scheme, G)
    excitatory_connections, inhibitory_connections = connection_builder.build(
        groups, neuron)

    stdp = ModelSTDPBuilder(config['model']).build(
        inhibitory_connections, connection_builder.g_I_bar)

    recording_duration = quantity(config['monitoring']['recording_duration'])
    M_E = b.RecentStateMonitor(neuron, 'I_E', recording_duration)
    M_I = b.RecentStateMonitor(neuron, 'I_I', recording_duration)
    M_spikes = b.SpikeMonitor(neuron)
    M_rates = b.PopulationRateMonitor(neuron, 1.0 * b.second)
    M_weights = b.StateMonitor(inhibitory_connections, 'w', record=True)
    M_excitatory_spikes = b.SpikeMonitor(groups.excitatory, record=True)
    M_inhibitory_spikes = b.SpikeMonitor(groups.inhibitory, record=True)

    net = b.Network(
        neuron, G, excitatory_connections, inhibitory_connections, stdp, M_E,
        M_I, M_spikes, M_rates, M_weights)

    with tables.openFile(args.output[0], 'w') as outfile:
        outfile.setNodeAttr('/', 'config', config)

        currents_group = outfile.createGroup(
            '/', 'currents', "Recorded currents")

        time_passed = 0 * b.second
        store_times = quantity_list(config['monitoring']['store_times'])
        for i, time in enumerate(store_times):
            print i
            net.run(time - time_passed)
            time_passed = time
            store_group = outfile.createGroup(currents_group, 't' + str(i))
            times_arr = outfile.createArray(store_group, 'times', M_E.times)
            times_arr.attrs.unit = "second"
            exc_arr = outfile.createArray(store_group, 'excitatory', M_E[0])
            exc_arr.attrs.unit = "amp"
            inh_arr = outfile.createArray(store_group, 'inhibitory', M_I[0])
            inh_arr.attrs.unit = "amp"
            outfile.flush()

        spike_arr = outfile.createArray(
            '/', 'spikes', M_spikes[0], "Spike times of model neuron.")
        spike_arr.attrs.unit = "second"
        outfile.flush()

        input_group = outfile.createGroup('/', 'input', "Incoming spikes.")
        exc_spikes_arr = outfile.createArray(
            input_group, 'excitatory', np.asarray(M_excitatory_spikes.spikes))
        exc_spikes_arr.attrs.unit = "second"
        inh_spikes_arr = outfile.createArray(
            input_group, 'inhibitory', np.asarray(M_inhibitory_spikes.spikes))
        inh_spikes_arr.attrs.unit = "second"
        outfile.createArray(
            input_group, 'exc_group_membership', groups.exc_group_membership,
            "Indicates which excitatory neuron belongs to which group.")
        outfile.createArray(
            input_group, 'inh_group_membership', groups.inh_group_membership,
            "Indicates which inhibitory neuron belongs to which group.")
        outfile.flush()

        rates_group = outfile.createGroup('/', 'rates', "Firing rates.")
        rates_arr = outfile.createArray(rates_group, 'rates', M_rates.rate)
        rates_arr.attrs.unit = "hertz"
        times_arr = outfile.createArray(
            rates_group, 'times', M_rates.times,
            "Times of the firing rate estimation bins.")
        times_arr.attrs.unit = "second"
        outfile.flush()

        weights_group = outfile.createGroup('/', 'weights', "Synaptic weights.")
        weights_arr = outfile.createArray(weights_group, 'weights', M_weights)
        weights_arr.attrs.unit = "siemens"
        times_arr = outfile.createArray(
            weights_group, 'times', M_weights.times,
            "Times of the recorded synaptic weights.")
        times_arr.attrs.unit = "second"
        outfile.flush()
