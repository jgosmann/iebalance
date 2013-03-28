#!/usr/bin/env python

from config import Configurable, EquationString, quantity, quantity_list
import brian as b
import inputs
import logging
import numpy as np
import numpy.random as rnd


logger = logging.getLogger('single-cell')


class ModelBuilder(Configurable):
    def __init__(self, config):
        Configurable.__init__(self, config)
        self._add_config_value('tau', quantity)  # membrane time constant
        self._add_config_value('V_rest', quantity)  # resting membrane potential
        self._add_config_value('threshold', quantity)  # spiking threshold
        self._add_config_value('g_leak', quantity)  # leak conductance
        self._add_config_value('refractory_period', quantity)

        self._add_config_value('V_exc', quantity)  # excitatory reversal potential
        self._add_config_value('V_inh', quantity)  # inhibitory reversal potential
        self._add_config_value('I_b', quantity)  # additional current

        self._add_config_value('g_exc_bar', quantity)
        self._add_config_value('g_inh_bar', quantity)
        self._add_config_value('tau_exc', quantity)  # excitat. syn. time constant
        self._add_config_value('tau_inh', quantity)  # inhibit. syn. time constant
        self._add_config_value('tau_stdp', quantity)
        self._add_config_value('eta', float)
        self._add_config_value('rho', quantity)

        self.alpha = 2 * self.rho * self.tau_stdp

        self.eqs = b.Equations('''
            I_exc : amp
            I_inh : amp
            dV/dt = ((self.V_rest - V) + (I_exc + I_inh + self.I_b) / \
                self.g_leak) / self.tau : volt
            dx/dt = -x / self.tau_stdp : 1
            ''')
        self.eqs_inh_synapse = SynapsesEquations(
            config['synapses']['inhibitory'])
        self.eqs_exc_synapse = '''
            dg/dt = -g / self.tau_exc : siemens
            I = g * (self.V_exc - V_post) : amp
            w : 1
            '''

    def build_neuron_group(self, num_neurons=1):
        return b.NeuronGroup(
            num_neurons, model=self.eqs, reset=self.V_rest,
            threshold=self.threshold, refractory=self.refractory_period)

    def build_exc_synapses(self, source, target, tuning):
        synapses = b.Synapses(
            source, target, model=self.eqs_exc_synapse, pre='g += w')
        synapses[:, :] = True
        synapses.w[:, :] = np.atleast_2d(self.g_exc_bar * tuning).T
        target.I_exc = synapses.I
        return synapses

    def build_inh_synapses(self, source, target):
        alpha = self.alpha
        eta = self.eta
        g_inh_bar = self.g_inh_bar
        tau_inh = self.tau_inh
        tau_stdp = self.tau_stdp
        E = self.V_inh
        exp = np.exp
        # suppress unused warnings
        assert alpha and eta and g_inh_bar and tau_inh and tau_stdp and E

        synapses = b.Synapses(
            source, target, model=self.eqs_inh_synapse.equations,
            pre=self.eqs_inh_synapse.pre, post=self.eqs_inh_synapse.post)
        synapses[:, :] = True
        synapses.w = 0.1 * g_inh_bar
        target.I_inh = synapses.I
        return synapses


class SynapsesEquations(Configurable):
    def __init__(self, config):
        Configurable.__init__(self, config)
        self._add_config_value('equations', EquationString('\n'))
        self._add_config_value('pre', EquationString('; '))
        self._add_config_value('post', EquationString('; '))


class ModelInputGroups(object):
    def __init__(self, indexing, input_group):
        for key in indexing.iterkeys():
            assert np.all(1 == np.diff(
                [idx for group in indexing[key] for idx in group]))
            assert np.all(0 == np.diff([len(group) in group in indexing[key]]))

        self.excitatory = input_group.subgroup(
            sum(len(group) for group in indexing['excitatory']))
        self.inhibitory = input_group.subgroup(
            sum(len(group) for group in indexing['inhibitory']))

        self.exc_group_membership = np.hstack(
            np.repeat(i, len(group))
            for i, group in enumerate(indexing['excitatory']))
        self.inh_group_membership = np.hstack(
            np.repeat(i, len(group))
            for i, group in enumerate(indexing['inhibitory']))


class SingleCellModel(b.Network):
    def __init__(self, config):
        b.Network.__init__(self)

        builder = ModelBuilder(config['model'])
        self.input_gen = inputs.GroupedSpikeTimesGenerator(config['inputs'])
        self.indexing_scheme = self.input_gen.get_indexing_scheme()

        self.neuron = builder.build_neuron_group()
        self.input_neurons = b.SpikeGeneratorGroup(
            self.input_gen.num_trains, inputs.swap_tuple_values(self.input_gen))
        self.input_groups = ModelInputGroups(
            self.indexing_scheme, self.input_neurons)
        self.exc_synapses = builder.build_exc_synapses(
            self.input_groups.excitatory, self.neuron,
            self.tuning_function(self.input_groups.exc_group_membership, 5))
        self.inh_synapses = builder.build_inh_synapses(
            self.input_groups.inhibitory, self.neuron)

        self.add(
            self.neuron, self.input_neurons, self.inh_synapses,
            self.exc_synapses)

    @staticmethod
    def tuning_function(subgroup_indices, peak):
        return 0.3 + rnd.rand(*subgroup_indices.shape) / 10.0 + \
            1.1 / (1.0 + (subgroup_indices + 1 - peak) ** 4)


class SingleCellModelRecorder(Configurable):
    def __init__(self, config, model):
        Configurable.__init__(self, config)
        self._add_config_value('recording_duration', quantity)
        self._add_config_value('rate_bin_size', quantity)
        self._add_config_value('store_times', quantity_list)
        self._add_config_value('current_timestep', int)
        self._add_config_value('weights_timestep', int)
        self.model = model

        self.m_spikes = b.SpikeMonitor(model.neuron)
        self.m_rates = b.PopulationRateMonitor(model.neuron, self.rate_bin_size)
        self.m_exc_syn_currents = b.RecentStateMonitor(
            model.exc_synapses, 'I', self.recording_duration,
            timestep=self.current_timestep)
        self.m_inh_syn_currents = b.RecentStateMonitor(
            model.inh_synapses, 'I', self.recording_duration,
            timestep=self.current_timestep)
        self.m_inh_weights = b.StateMonitor(
            model.inh_synapses, 'w', record=True,
            timestep=self.weights_timestep)

        self.model.add(
            self.m_spikes, self.m_rates, self.m_exc_syn_currents,
            self.m_inh_syn_currents, self.m_inh_weights)

    def record(self, outfile):
        self._store_group_memberships(outfile)
        outfile.flush()

        time_passed = 0 * b.second
        for i, time in enumerate(self.store_times):
            logger.info(
                'Running time interval %i (duration %ds, end time %ds)',
                i, time - time_passed, time)
            self.model.run(time - time_passed, report='text')
            time_passed = time
            self._store_recent_currents(outfile, i)
            outfile.flush()

        self._store_rates(outfile)
        self._store_spikes(outfile)
        self._store_weights(outfile)
        outfile.flush()

    @staticmethod
    def _store_array_with_unit(
            outfile, where, name, array, unit, *args, **kwargs):
        node = outfile.createArray(where, name, array, *args, **kwargs)
        node.attrs.unit = unit
        return node

    def _store_rates(self, outfile):
        group = outfile.createGroup('/', 'rates', "Firing rates.")
        self._store_array_with_unit(
            outfile, group, 'rates', self.m_rates.rate / b.hertz, 'hertz')
        self._store_array_with_unit(
            outfile, group, 'times', self.m_rates.times / b.second, 'second',
            "Times of the firing rate estimation bins.")

    def _store_group_memberships(self, outfile):
        group = outfile.createGroup('/', 'group_memberships')
        outfile.createArray(
            group, 'inhibitory', self.model.input_groups.inh_group_membership)
        outfile.createArray(
            group, 'excitatory', self.model.input_groups.exc_group_membership)

    def _store_recent_currents(self, outfile, interval_index):
        from numpy.testing import assert_allclose
        assert_allclose(
            self.m_exc_syn_currents.times, self.m_inh_syn_currents.times)

        group = outfile.createGroup(
            '/currents', 't' + str(interval_index), createparents=True)
        self._store_array_with_unit(
            outfile, group, 'times', self.m_exc_syn_currents.times / b.second,
            'second')
        self._store_array_with_unit(
            outfile, group, 'excitatory',
            self.m_exc_syn_currents.values / b.amp, 'amp')
        self._store_array_with_unit(
            outfile, group, 'inhibitory',
            self.m_inh_syn_currents.values / b.amp, 'amp')

    def _store_spikes(self, outfile):
        self._store_array_with_unit(
            outfile, '/', 'spikes', self.m_spikes[0], 'second'
            "Spike times of model neuron.")

    def _store_weights(self, outfile):
        weight_group = outfile.createGroup('/', 'weights', "Synaptic weights.")
        group = outfile.createGroup(weight_group, 'inhibitory')
        self._store_array_with_unit(
            outfile, group, 'weights', self.m_inh_weights.values / b.siemens,
            'siemens')
        self._store_array_with_unit(
            outfile, group, 'times', self.m_inh_weights.times / b.second,
            'second', "Times of the recorded synaptic weights.")


if __name__ == '__main__':
    import argparse
    import json
    import tables
    from brian.globalprefs import set_global_preferences
    set_global_preferences(useweave=True)

    logging.basicConfig()
    logger.setLevel(logging.INFO)

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

    b.defaultclock.dt = quantity(config['dt'])
    model = SingleCellModel(config)
    recorder = SingleCellModelRecorder(config['recording'], model)

    with tables.openFile(args.output[0], 'w') as outfile:
        outfile.setNodeAttr('/', 'config', config)
        recorder.record(outfile)
