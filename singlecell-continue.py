#!/usr/bin/env python

from config import Configurable, quantity, quantity_list
import brian as b
import logging
import numpy as np
from singlecell import SingleCellModel


logger = logging.getLogger('single-cell-continue')


class SingleCellModelContinuedRecorder(Configurable):
    def __init__(self, input_data, model):
        Configurable.__init__(
            self, input_data.getNodeAttr('/', 'config')['recording'])
        self._add_config_value('recording_duration', quantity)
        self._add_config_value('rate_bin_size', quantity)
        self._add_config_value('store_times', quantity_list)
        self._add_config_value('current_timestep', int)
        self._add_config_value('weights_timestep', int)
        self.input_data = input_data
        self.time_passed = input_data.root.weights.inhibitory.times[-1]
        self.model = model

        self.m_spikes = b.SpikeMonitor(model.neuron)
        self.m_rates = b.PopulationRateMonitor(model.neuron, self.rate_bin_size)
        self.m_exc_syn_currents = b.RecentStateMonitor(
            model.neuron, 'I_exc', self.recording_duration,
            timestep=self.current_timestep)
        self.m_inh_syn_currents = b.RecentStateMonitor(
            model.neuron, 'I_inh', self.recording_duration,
            timestep=self.current_timestep)
        self.m_exc_weights = b.StateMonitor(
            model.exc_synapses, 'w', record=True,
            timestep=self.weights_timestep)
        self.m_inh_weights = b.StateMonitor(
            model.inh_synapses, 'w', record=True,
            timestep=self.weights_timestep)

        self.model.add(
            self.m_spikes, self.m_rates, self.m_exc_syn_currents,
            self.m_inh_syn_currents, self.m_exc_weights, self.m_inh_weights)

    def record(self, outfile, time):
        self._store_group_memberships(outfile)
        outfile.flush()

        logger.info('Running time interval of duration %ds up to %ds' % (
            time, time / b.second + self.time_passed))
        self.model.run(time, report='text')
        logger.info('Storing data')
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
            outfile, group, 'rates',
            np.hstack((self.input_data.root.rates.rates,
                       self.m_rates.rate / b.hertz)),
            'hertz')
        self._store_array_with_unit(
            outfile, group, 'times',
            np.hstack((self.input_data.root.rates.times,
                       self.time_passed + self.m_rates.times / b.second)),
            'second', "Times of the firing rate estimation bins.")

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
            outfile, group, 'times',
            self.time_passed + self.m_exc_syn_currents.times / b.second,
            'second')
        self._store_array_with_unit(
            outfile, group, 'excitatory',
            self.m_exc_syn_currents.values / b.amp, 'amp')
        self._store_array_with_unit(
            outfile, group, 'inhibitory',
            self.m_inh_syn_currents.values / b.amp, 'amp')

    def _store_spikes(self, outfile):
        self._store_array_with_unit(
            outfile, '/', 'spikes',
            np.hstack((self.input_data.root.spikes,
                       self.time_passed + self.m_spikes[0])),
            'second', "Spike times of model neuron.")

    def _store_weights(self, outfile):
        weight_group = outfile.createGroup('/', 'weights', "Synaptic weights.")
        group = outfile.createGroup(weight_group, 'inhibitory')
# FIXME
#100% complete, 15m 19s elapsed, approximately 0s remaining.
#INFO:single-cell-continue:Storing data
#Traceback (most recent call last):
  #File "singlecell-continue.py", line 177, in <module>
    #recorder.record(outfile, args.time[0] * b.second)
  #File "singlecell-continue.py", line 54, in record
    #self._store_weights(outfile)
  #File "singlecell-continue.py", line 113, in _store_weights
    #self.m_inh_weights.values / b.siemens)),
  #File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/shape_base.py", line 226, in vstack
    #return _nx.concatenate(map(atleast_2d,tup),0)
#ValueError: all the input array dimensions except for the concatenation axis must match exactly
        self._store_array_with_unit(
            outfile, group, 'weights',
            np.hstack((self.input_data.root.weights.inhibitory.weights,
                       self.m_inh_weights.values / b.siemens)),
            'siemens')
        self._store_array_with_unit(
            outfile, group, 'times',
            np.hstack((self.input_data.root.weights.inhibitory.times,
                       self.time_passed + self.m_inh_weights.times / b.second)),
            'second', "Times of the recorded synaptic weights.")
        group = outfile.createGroup(weight_group, 'excitatory')
        self._store_array_with_unit(
            outfile, group, 'weights',
            np.hstack((self.input_data.root.weights.excitatory.weights,
                       self.m_exc_weights.values / b.siemens)),
            'siemens')
        self._store_array_with_unit(
            outfile, group, 'times',
            np.hstack((self.input_data.root.weights.excitatory.times,
                       self.time_passed + self.m_exc_weights.times / b.second)),
            'second', "Times of the recorded synaptic weights.")


if __name__ == '__main__':
    import argparse
    import os.path
    import tables
    from brian.globalprefs import set_global_preferences
    set_global_preferences(useweave=True)

    logging.basicConfig()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run the Vogels et al. 2011 single cell model.")
    parser.add_argument(
        '-i', '--input', type=str, nargs=1, required=True,
        help="Path to the input file for which to continue the simulation.")
    parser.add_argument(
        '-t', '--time', type=float, nargs=1, required=True,
        help="Additional time to simulate in seconds.")
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

    with tables.openFile(args.input[0], 'r') as input_data:
        config = input_data.getNodeAttr('/', 'config')
        b.defaultclock.dt = quantity(config['dt'])
        model = SingleCellModel(config)
        model.exc_synapses.w[:, :] = \
            input_data.root.weights.excitatory.weights[:, -1]
        model.inh_synapses.w[:, :] = \
            input_data.root.weights.inhibitory.weights[:, -1]
        recorder = SingleCellModelContinuedRecorder(input_data, model)
        # FIXME update store times? no
        # FIXME set correct time done
        #b.defaultclock.reinit(
            #quantity_list(config['recording']['store_times'])[-1])
        # FIXME: Set weights correctly done
        # FIXME: What takes so long?

        with tables.openFile(os.path.join(outpath, args.output[0]), 'w') as outfile:
            outfile.setNodeAttr('/', 'config', config)
            recorder.record(outfile, args.time[0] * b.second)
