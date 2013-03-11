#!/usr/bin/env python

from joblib import Parallel, delayed
import brian
import brian.units
import itertools
import logging
import numpy as np
import numpy.random as rnd
import quantities as pq
import spykeutils.spike_train_generation as stg
import tables


logger = logging.getLogger('inputs')


def quantity(l):
    magnitude, unit = l
    return magnitude * getattr(brian.units, unit)


class Configurable(object):
    def __init__(self, config):
        self._config = config

    def _add_config_value(self, name, handler=None, default=None):
        try:
            value = self._config[name]
        except KeyError as e:
            if default is not None:
                value = default
            else:
                raise e
        if handler is not None:
            value = handler(value)
        setattr(self, name, value)


class InputSignalGenerator(Configurable):
    def __init__(self, config):
        Configurable.__init__(self, config)
        self._add_config_value('length', quantity)
        self._add_config_value('peak_firing_rate', quantity)
        self._add_config_value('background_activity', quantity)
        self._add_config_value('sparseness', int)
        self._add_config_value('filter_time_constant', quantity)
        self._add_config_value('dt', quantity)
        self.size = int(self.length / self.dt)

    def gen_filtered_white_noise(self):
        r = rnd.rand(self.size) - 0.5
        filter_value = np.exp(-self.dt / self.filter_time_constant)
        signal = np.empty(self.size)
        signal[0] = (1 - filter_value) * r[0]
        for i in xrange(1, signal.size):
            signal[i] = (1 - filter_value) * r[i] + filter_value * signal[i - 1]
        return signal

    @staticmethod
    def remove_bumps(signal, n):
        bump_borders = np.diff(np.asarray(signal > 0, dtype=int))
        bump_starts = np.nonzero(bump_borders == 1)[0]
        bump_ends = np.nonzero(bump_borders == -1)[0]
        if bump_starts.size > 0 and bump_ends.size > 0:
            while bump_starts[0] > bump_ends[0]:
                bump_ends = bump_ends[1:]
        for i in xrange(1, bump_starts.size, n):
            start = bump_starts[i] + 1
            if i < bump_ends.size:
                end = bump_ends[i] + 1
            else:
                end = bump_ends.size
            signal[start:end] = 0
        return signal

    def rectify(self, signal):
        sparsified = self.remove_bumps(signal, self.sparseness)
        rectified = np.maximum(0, sparsified)
        normalized = self.peak_firing_rate * self.dt * rectified / \
            rectified.max()
        return normalized + self.background_activity * self.dt


def gen_trains_for_tuning(generator, i):
    return generator.gen_trains_for_tuning(i)


class SpikeTimesGenerator(Configurable):
    def __init__(self, config):
        Configurable.__init__(self, config)
        self.signal_gen = InputSignalGenerator(self._config['raw_signals'])

        self._add_config_value('num_tunings', int)
        self._add_config_value('num_trains', int)
        self._add_config_value('fraction_inhibitory', float)
        self._add_config_value('refractory_period', quantity)

        self.num_trains_per_tuning = self.num_trains // self.num_tunings
        self.num_inhib_per_tuning = \
            int(self.fraction_inhibitory * self.num_trains_per_tuning)
        self.num_excit_per_tuning = \
            int(self.num_trains_per_tuning - self.num_inhib_per_tuning)

    def gen_spike_train(self, input_signal):
        max_input = input_signal.max()
        max_rate = max_input / self.signal_gen.dt

        def modulation(ts):
            return np.interp(
                ts.rescale(pq.ms).magnitude,
                np.arange(input_signal.size) * self.signal_gen.dt,
                input_signal) / max_input

        return stg.gen_inhomogeneous_poisson(
            modulation, max_rate / brian.hertz * pq.Hz,
            t_stop=self.signal_gen.length / brian.second * pq.s,
            refractory=self.refractory_period / brian.second * pq.s)

    @staticmethod
    def trains_to_spiketimes_list(trains):
        spikes = []
        for i, train in enumerate(trains):
            spikes.extend((i, spike.rescale(pq.s).magnitude) for spike in train)
        spikes.sort(key=lambda (i, t): t)
        return spikes

    def gen_trains_for_tuning(self, i):
        logger.info("Generating spike trains for tuning %i of %i ..." %
                    (i + 1, self.num_tunings))
        raw_signal = self.signal_gen.gen_filtered_white_noise()
        input_signal = self.signal_gen.rectify(raw_signal)
        excitatory = [self.gen_spike_train(input_signal)
                      for i in xrange(self.num_excit_per_tuning)]
        inhibitory = [self.gen_spike_train(input_signal)
                      for i in xrange(self.num_inhib_per_tuning)]
        return excitatory, inhibitory, raw_signal[-1]

    def generate(self, num_jobs=None, verbose=0):
        if num_jobs is not None:
            num_jobs = min(self.num_tunings, num_jobs)
        trains = Parallel(num_jobs, verbose)(delayed(gen_trains_for_tuning)(
            self, i) for i in xrange(self.num_tunings))
        excitatory, inhibitory, last_raw_signal_values = zip(*trains)
        return self.trains_to_spiketimes_list(itertools.chain(
            itertools.chain(*excitatory), itertools.chain(*inhibitory))), \
            last_raw_signal_values

    def get_indexing_scheme(self):
        num_excitatory = self.num_tunings * self.num_excit_per_tuning
        return {
            'excitatory': [np.arange(
                i * self.num_excit_per_tuning,
                (i + 1) * self.num_excit_per_tuning)
                for i in xrange(self.num_tunings)],
            'inhbitory': [num_excitatory + np.arange(
                i * self.num_inhib_per_tuning,
                (i + 1) * self.num_inhib_per_tuning)
                for i in xrange(self.num_tunings)]
        }


class SpiketimesTable(tables.IsDescription):
    neuron_index = tables.UIntCol()
    spiketime = tables.Float32Col()


if __name__ == '__main__':
    import argparse
    import json

    logging.basicConfig()

    parser = argparse.ArgumentParser(
        description="Produce input spike times for the Vogels et al. 2011 " +
        "model.")
    parser.add_argument(
        '-c', '--config', type=str, nargs=1, required=True,
        help="Path to the configuration file.")
    parser.add_argument(
        'output', nargs=1, type=str,
        help="Path ot the HDF5 output file.")
    parser.add_argument(
        '-j', '--jobs', nargs=1, type=int, default=[1],
        help="Number of parallel jobs to use. If a negative number n is " +
        "passed -n - 1 cores will be left unused. Default value is 1.")
    parser.add_argument(
        '-v', '--verbose', nargs='?', type=int, const=0, default=-1,
        help="Enable verbosity. Accepts an integer to control level of " +
        "verbosity. Use -1 to switch off.")
    args = parser.parse_args()

    if args.verbose >= 0:
        logger.setLevel(logging.INFO)

    with open(args.config[0], 'r') as f:
        config = json.load(f)

    with tables.openFile(args.output[0], 'w') as out_file:
        input_group = out_file.createGroup('/', 'input_data', "Input data")
        spike_table = out_file.createTable(
            input_group, 'spiketimes', SpiketimesTable,
            "Table of spike times of Poisson spike train and the index of the" +
            " neuron producing the spike.")
        generator = SpikeTimesGenerator(config)
        spiketimes, last_raw_signal_values = generator.generate(
            args.jobs[0], args.verbose)
        spike_table.append(spiketimes)
        spike_table.attrs.spiketime_unit = 'second'
        spike_table.attrs.config = config
        out_file.flush()

        raw_signal_array = out_file.createArray(
            input_group, 'last_raw_signal_values', last_raw_signal_values,
            "The last raw signal value of each tuning group. Can be used to " +
            "extend the length of the input signals later on.")

        indexing_group = out_file.createGroup(
            input_group, 'indexing', "Indices of excitatory and inhibitory " +
            "neurons for the different tuning groups.")
        for key, groups in generator.get_indexing_scheme().iteritems():
            for i, group in enumerate(groups):
                path = '%s/%s' % (indexing_group._v_pathname, key)
                out_file.createArray(
                    path, 'group%i' % i, group, createparents=True)
        out_file.flush()
