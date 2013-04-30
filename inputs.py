#!/usr/bin/env python

from config import Configurable, quantity
import brian
import heapq
import numpy as np
import numpy.random as rnd
import quantities as pq
import spykeutils.spike_train_generation as stg
import tables


class InputSignalGenerator(Configurable):
    def __init__(self, config, duration=None):
        Configurable.__init__(self, config)
        self._add_config_value('peak_firing_rate', quantity)
        self._add_config_value('background_activity', quantity)
        self._add_config_value('sparseness', int)
        self._add_config_value('approximate_normalization', float)
        self._add_config_value('filter_time_constant', quantity)
        self._add_config_value('dt', quantity)
        self.current_raw_value = 0.0
        self.current_time = 0.0 * brian.second
        self.sparsification_start = 1
        self.signal = None
        self.duration = duration

    def gen_filtered_white_noise(self, initial_value, size):
        r = rnd.rand(size) - 0.5
        filter_value = np.exp(-self.dt / self.filter_time_constant)
        signal = np.empty(size)
        signal[0] = (1 - filter_value) * r[0] + filter_value * initial_value
        for i in xrange(1, signal.size):
            signal[i] = (1 - filter_value) * r[i] + filter_value * signal[i - 1]
        return signal

    def next_interval(self):
        if self.duration is None:
            duration = 4 * brian.second
            self.signal = None
        else:
            duration = self.duration
        if self.signal is None:
            self.signal = self.gen_filtered_white_noise(
                self.current_raw_value, duration / self.dt)
            self.current_raw_value = self.signal[-1]
            self.rectify(self.signal)
        self.current_time += duration
        return self.signal

    def rectify(self, signal):
        self.remove_bumps(signal)
        np.maximum(0, signal, out=signal)
        signal *= self.peak_firing_rate * self.dt / \
            self.approximate_normalization
        signal += self.background_activity * self.dt

    def remove_bumps(self, signal):
        start = self.sparsification_start
        bump_borders = np.diff(np.asarray(signal > 0, dtype=int))
        bump_starts = np.nonzero(bump_borders == 1)[0]
        bump_ends = np.nonzero(bump_borders == -1)[0]
        while bump_starts.size > 0 and bump_ends.size > 0 \
                and bump_starts[0] > bump_ends[0]:
            if start <= 0:
                signal[0:bump_ends[0] + 1] = 0
                start += self.sparseness - 1
            else:
                start = start - 1
            bump_ends = bump_ends[1:]
        self.sparsification_start = self.sparseness - 1 - \
            (bump_starts.size - start) % self.sparseness
        for i in xrange(start, bump_starts.size, self.sparseness):
            bump_start = bump_starts[i] + 1
            if i < bump_ends.size:
                bump_end = bump_ends[i] + 1
            else:
                bump_end = bump_ends.size
                self.sparsification_start = 0
            signal[bump_start:bump_end] = 0


class PoissonSpikeTimesGenerator(object):
    def __init__(self, signal_gen, neuron_indices, refractory_period):
        self.signal_gen = signal_gen
        self.neuron_indices = neuron_indices
        self.refractory_period = refractory_period
        self.last_spikes = np.empty(neuron_indices.size)
        self.last_spikes.fill(-self.refractory_period)
        self.__spikes = []
        self.__current_index = -1

    def __iter__(self):
        return self

    def __fill_spikes_for_next_interval(self):
        t_starts = np.maximum(
            self.signal_gen.current_time,
            self.last_spikes + self.refractory_period)
        signal = self.signal_gen.next_interval()
        t_stop = self.signal_gen.current_time
        trains = [self.gen_spike_train(signal, t_starts[i], t_stop).rescale(
            pq.s).magnitude for i in xrange(len(self.neuron_indices))]
        self.last_spikes = np.asarray(
            [st[-1] if st.size > 0 else self.last_spikes[i]
             for i, st in enumerate(trains)]) * brian.second
        self.__spikes = sorted(
            (st * brian.second, i)
            for i, train in zip(self.neuron_indices, trains) for st in train)

    def gen_spike_train(self, input_signal, t_start, t_stop):
        t_start /= brian.second / pq.s
        t_stop /= brian.second / pq.s
        max_input = input_signal.max()
        dt = self.signal_gen.dt
        max_rate = max_input / self.signal_gen.dt
        input_signal_duration = input_signal.size * dt
        input_signal_times = self.signal_gen.current_time - \
            input_signal_duration + np.arange(input_signal.size) * dt

        def modulation(ts):
            return np.interp(
                ts.rescale(pq.s).magnitude, input_signal_times,
                input_signal) / max_input

        return stg.gen_inhomogeneous_poisson(
            modulation, max_rate / brian.hertz * pq.Hz,
            t_start=t_start, t_stop=t_stop,
            refractory=self.refractory_period / brian.second * pq.s)

    def next(self):
        self.__current_index += 1
        if self.__current_index >= len(self.__spikes):
            self.__fill_spikes_for_next_interval()
            self.__current_index = 0
        return self.__spikes[self.__current_index]


class GroupedSpikeTimesGenerator(Configurable):
    def __init__(self, config, duration=None):
        Configurable.__init__(self, config)

        self._add_config_value('num_tunings', int)
        self._add_config_value('num_trains', int)
        self._add_config_value('fraction_inhibitory', float)
        self._add_config_value('refractory_period', quantity)

        self.num_trains_per_tuning = self.num_trains // self.num_tunings
        self.num_inhibitory = int(self.fraction_inhibitory * self.num_trains)
        self.num_excitatory = self.num_trains - self.num_inhibitory
        self.num_inhib_per_tuning = self.num_inhibitory / self.num_tunings
        self.num_excit_per_tuning = self.num_excitatory / self.num_tunings

        self.signal_gens = [
            InputSignalGenerator(self._config['raw_signals'], duration)
            for i in xrange(self.num_tunings)]
        time_gens = [PoissonSpikeTimesGenerator(
            gen, self.neuron_indices_of_group(i), self.refractory_period)
            for i, gen in enumerate(self.signal_gens)]
        self.merged_times = heapq.merge(*time_gens)

    def __iter__(self):
        return self.merged_times

    def excitatory_neuron_indices_of_group(self, n):
        start = n * self.num_excit_per_tuning
        return np.arange(start, start + self.num_excit_per_tuning)

    def inhibitory_neuron_indices_of_group(self, n):
        start = n * self.num_inhib_per_tuning + self.num_excitatory
        return np.arange(start, start + self.num_inhib_per_tuning)

    def neuron_indices_of_group(self, n):
        return np.hstack((
            self.excitatory_neuron_indices_of_group(n),
            self.inhibitory_neuron_indices_of_group(n)))

    def get_indexing_scheme(self):
        return {
            'excitatory': [self.excitatory_neuron_indices_of_group(i)
                           for i in xrange(self.num_tunings)],
            'inhibitory': [self.inhibitory_neuron_indices_of_group(i)
                           for i in xrange(self.num_tunings)]
        }


def swap_tuple_values(tuples):
    for x, y in tuples:
        yield y, x


class SpiketimesTable(tables.IsDescription):
    neuron_index = tables.UIntCol()
    spiketime = tables.Float32Col()


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Produce input spike times for the Vogels et al. 2011 " +
        "model.")
    parser.add_argument(
        '-c', '--config', type=str, nargs=1, required=True,
        help="Path to the configuration file.")
    parser.add_argument(
        '-t', '--time', type=float, nargs=1, required=True,
        help="Time point in seconds up to which to generate spikes.")
    parser.add_argument(
        'output', nargs=1, type=str,
        help="Path ot the HDF5 output file.")
    parser.add_argument(
        'label', nargs='?', type=str,
        help="Label for the simulation. Not used!")
    args = parser.parse_args()

    with open(args.config[0], 'r') as f:
        config = json.load(f)

    with tables.openFile(args.output[0], 'w') as out_file:
        out_file.setNodeAttr('/', 'config', config)
        input_group = out_file.createGroup('/', 'input_data', "Input data")
        spike_table = out_file.createTable(
            input_group, 'spiketimes', SpiketimesTable,
            "Table of spike times of Poisson spike train and the index of the" +
            " neuron producing the spike.")
        spike_table.attrs.spiketime_unit = 'second'

        generator = GroupedSpikeTimesGenerator(config)
        geniter = iter(generator)
        t = 0 * brian.second
        i = 0
        while t < args.time[0] * brian.second:
            t, idx = geniter.next()
            spike = spike_table.row
            spike['neuron_index'] = idx
            spike['spiketime'] = t
            spike.append()
            i += 1
            if i % 1000:
                spike_table.flush()
        out_file.flush()

        indexing_group = out_file.createGroup(
            input_group, 'indexing', "Indices of excitatory and inhibitory " +
            "neurons for the different tuning groups.")
        for key, groups in generator.get_indexing_scheme().iteritems():
            for i, group in enumerate(groups):
                path = '%s/%s' % (indexing_group._v_pathname, key)
                out_file.createArray(
                    path, 'group%i' % i, group, createparents=True)
        out_file.flush()
