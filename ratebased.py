#!/usr/bin/env python

from config import Configurable, quantity
from inputs import InputSignalGenerator
import brian as b
import logging
import numpy as np
import tables


logger = logging.getLogger('ratebased')


class RateModel(Configurable):
    def __init__(self, config, outfile):
        Configurable.__init__(self, config)
        self._add_config_value('epoch_duration', quantity)
        self._add_config_value('num_tunings', int)
        self._add_config_value('num_epochs', int)
        self._add_config_value('eta', float)
        self._add_config_value('rho', quantity)

        self.outfile = outfile

        self.exc_weights = 0.5 * (0.35 + 1.1 / (
            1.0 + (np.arange(self.num_tunings) - 4) ** 4))
        self.inh_weights = np.empty(self.num_tunings)
        self.inh_weights.fill(0.05)
        self.signal_gens = [InputSignalGenerator(
            self._config['raw_signals'], self.epoch_duration)
            for i in xrange(self.num_tunings)]

    def run_epoch(self):
        signals = np.asarray([gen.next_interval() for gen in self.signal_gens])
        for t in xrange(signals.shape[1]):
            exc = np.dot(signals[:, t], self.exc_weights)
            inh = np.dot(signals[:, t], self.inh_weights)
            self.post = min(150 * b.hertz * self.signal_gens[0].dt, max(0.0, exc - inh))
            dw = self.eta * signals[:, t] * (
                self.post - self.rho * self.signal_gens[0].dt)
            #dw = self.eta * (signals[:, t] * self.post - 0.001 * self.inh_weights)
            self.inh_weights += dw

    def run_for_corr(self):
        for signal_gen in self.signal_gens:
            signal_gen.duration = 20 * b.second
        signals = np.asarray([gen.next_interval() for gen in self.signal_gens])
        post = np.empty(signals.shape[1])
        for t in xrange(signals.shape[1]):
            exc = np.dot(signals[:, t], self.exc_weights)
            inh = np.dot(signals[:, t], self.inh_weights)
            post[t] = min(150 * b.hertz * self.signal_gens[0].dt, max(0.0, exc - inh))

        group = self.outfile.createGroup('/', 'signals')
        for i, sig in enumerate(signals):
            self.outfile.createArray(group, 's%s' % i, sig)
        self.outfile.createArray('/', 'trained_rates', post)

    def run(self):
        self._create_rates_arrays()
        self._create_weights_arrays()
        self._store_group_memberships()
        for i in xrange(self.num_epochs):
            self.run_epoch()
            t = i * self.epoch_duration
            self._store_rate(t, self.post / self.signal_gens[0].dt)
            self._store_weights(t, self.inh_weights)
        self.run_for_corr()

    def _create_rates_arrays(self):
        group = self.outfile.createGroup('/', 'rates', "Firing rates.")
        arr = self.outfile.createEArray(
            group, 'rates', tables.Float32Col(), (0,))
        arr.attrs.unit = 'hertz'
        arr = self.outfile.createEArray(
            group, 'times', tables.Float32Col(), (0,),
            "Times of the firing rate estimation bins.")
        arr.attrs.unit = 'second'

    def _create_weights_arrays(self):
        weight_group = self.outfile.createGroup(
            '/', 'weights', "Synaptic weights.")
        group = self.outfile.createGroup(weight_group, 'inhibitory')
        arr = self.outfile.createEArray(
            group, 'weights', tables.Float32Col(), (8, 0))
        arr.attrs.unit = ''
        arr = self.outfile.createEArray(
            group, 'times', tables.Float32Col(), (0,),
            "Times of the recorded synaptic weights.")
        arr.attrs.unit = 'second'
        group = outfile.createGroup(weight_group, 'excitatory')
        self.outfile.createArray(group, 'weights', self.exc_weights)

    def _store_group_memberships(self):
        group = self.outfile.createGroup('/', 'group_memberships')
        self.outfile.createArray(
            group, 'inhibitory', np.arange(self.num_tunings))
        self.outfile.createArray(
            group, 'excitatory', np.arange(self.num_tunings))

    def _store_rate(self, time, rate):
        self.outfile.root.rates.times.append(np.atleast_1d(time))
        self.outfile.root.rates.rates.append(np.atleast_1d(rate))

    def _store_weights(self, time, weights):
        self.outfile.root.weights.inhibitory.times.append(np.atleast_1d(time))
        self.outfile.root.weights.inhibitory.weights.append(
            np.atleast_2d(weights).T)


if __name__ == '__main__':
    import argparse
    import json
    import os.path

    logging.basicConfig()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run the Vogels et al. 2011 rate based single cell model.")
    parser.add_argument(
        '-c', '--config', type=str, nargs=1, required=True,
        help="Path to the configuration file.")
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

    with tables.openFile(os.path.join(outpath, args.output[0]), 'w') as outfile:
        outfile.setNodeAttr('/', 'config', config)
        model = RateModel(config, outfile)
        model.run()
