#!/usr/bin/env python

from config import Configurable, quantity
import brian as b
import inputs
import logging
from singlecell import ModelInputGroups


logger = logging.getLogger('single-cell-trained')


class TrainedModelBuilder(Configurable):
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
        self._add_config_value('init_inh_w', quantity)
        self._add_config_value('tau_exc', quantity)  # excitat. syn. time constant
        self._add_config_value('tau_inh', quantity)  # inhibit. syn. time constant
        self._add_config_value('tau_stdp', quantity)
        self._add_config_value('tau_w', quantity)
        self._add_config_value('eta', float)
        self._add_config_value('rho', quantity)

        self.alpha = 2 * self.rho * self.tau_stdp

        self.eqs = b.Equations('''
            dg_exc/dt = -g_exc / self.tau_exc : siemens
            I_exc = g_exc * (self.V_exc - V) : amp
            dg_inh/dt = -g_inh / self.tau_inh : siemens
            I_inh = g_inh * (self.V_exc - V) : amp
            dV/dt = ((self.V_rest - V) + (I_exc + I_inh + self.I_b) / \
                self.g_leak) / self.tau : volt
            dx/dt = -x / self.tau_stdp : 1
            ''')

    def build_neuron_group(self, num_neurons=1):
        return b.NeuronGroup(
            num_neurons, model=self.eqs, reset=self.V_rest,
            threshold=self.threshold, refractory=self.refractory_period)

    def build_exc_synapses(self, source, target, weights):
        connection = b.Connection(source, target, 'g_exc')
        connection[:, 0] = weights
        return connection

    def build_inh_synapses(self, source, target, weights):
        connection = b.Connection(source, target, 'g_inh')
        connection[:, 1] = weights
        return connection


class SingleCellTrainedModel(b.Network, Configurable):
    def __init__(self, config, exc_weights, inh_weights):
        b.Network.__init__(self)
        Configurable.__init__(self, config)
        self._add_config_value('stimulus_duration', quantity)

        builder = TrainedModelBuilder(config['model'])
        self.input_gen = inputs.GroupedSpikeTimesGenerator(
            config['inputs'], self.stimulus_duration)
        self.indexing_scheme = self.input_gen.get_indexing_scheme()

        self.neuron = builder.build_neuron_group()
        self.input_neurons = b.SpikeGeneratorGroup(
            self.input_gen.num_trains, inputs.swap_tuple_values(self.input_gen))
        self.input_groups = ModelInputGroups(
            self.indexing_scheme, self.input_neurons)
        self.exc_synapses = builder.build_exc_synapses(
            self.input_groups.excitatory, self.neuron, exc_weights)
        self.inh_synapses = builder.build_inh_synapses(
            self.input_groups.inhibitory, self.neuron, inh_weights)

        self.add(
            self.neuron, self.input_neurons, self.inh_synapses,
            self.exc_synapses)


class SingleCellModelSpikeRecorder(Configurable):
    def __init__(self, config, model):
        Configurable.__init__(self, config)
        self._add_config_value('stimulus_duration', quantity)
        self._add_config_value('num_trials', int)
        self.model = model

        self.m_spikes = b.SpikeMonitor(model.neuron)
        self.model.add(self.m_spikes)

    def record(self, outfile):
        self._store_group_memberships(outfile)
        outfile.flush()

        for i in xrange(self.num_trials):
            logger.info(
                'Running time trial %i of %i', i, self.num_trials)
            self.model.run(self.stimulus_duration, report='text')
            self._store_recent_spikes(outfile, i)
            outfile.flush()

    @staticmethod
    def _store_array_with_unit(
            outfile, where, name, array, unit, *args, **kwargs):
        node = outfile.createArray(where, name, array, *args, **kwargs)
        node.attrs.unit = unit
        return node

    def _store_group_memberships(self, outfile):
        group = outfile.createGroup('/', 'group_memberships')
        outfile.createArray(
            group, 'inhibitory', self.model.input_groups.inh_group_membership)
        outfile.createArray(
            group, 'excitatory', self.model.input_groups.exc_group_membership)

    def _store_recent_spikes(self, outfile, trial):
        self._store_array_with_unit(
            outfile, '/spikes', 'trial%1' % trial, self.m_spikes[0], 'second'
            "Spike times of model neuron.", createparents=True)
        self.m_spikes.reinit()


if __name__ == '__main__':
    import argparse
    import json
    import os.path
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
        '-i', '--input', type=str, nargs=1, required=True,
        help="Path to the data of the trained model used as input.")
    parser.add_argument(
        'output', nargs=1, type=str,
        help="Filename of the HDF5 output file.")
    parser.add_argument(
        'label', nargs='?', type=str,
        help="Label for the simulation. Will create a directory with the same "
        + "to store the produced data.")
    args = parser.parse_args()

    outpath = 'Data'
    if args.label is not None:
        outpath = os.path.join(outpath, args.label)

    with open(args.config[0], 'r') as f:
        config = json.load(f)

    b.defaultclock.dt = quantity(config['dt'])

    with tables.openFile(args.input[0], 'r') as data:
        model = SingleCellTrainedModel(
            config, data.root.weights.excitatory.weights,
            data.root.weights.inhibitory.weights[:, -1])
    recorder = SingleCellModelSpikeRecorder(config, model)

    with tables.openFile(os.path.join(outpath, args.output[0]), 'w') as outfile:
        outfile.setNodeAttr('/', 'config', config)
        recorder.record(outfile)
