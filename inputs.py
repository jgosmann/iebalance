import brian
import numpy as np
import numpy.random as rnd
import quantities as pq
import scipy.linalg
import spykeutils.spike_train_generation as stg


def gen_filtered_white_noise(size, dt, filter_time_constant):
    filter_value = np.exp(-dt / filter_time_constant)
    polynomials = np.repeat(filter_value, size)
    polynomials[0] = 1.0
    polynomials = np.cumprod(polynomials)
    return np.sum((1 - filter_value) * (rnd.rand(size) - 0.5) *
                  scipy.linalg.toeplitz(polynomials, np.zeros(size)), axis=1)


def remove_bumps(signal, n):
    bump_borders = np.diff(np.asarray(signal > 0, dtype=int))
    bump_starts = np.nonzero(bump_borders == 1)[0]
    bump_ends = np.nonzero(bump_borders == -1)[0]
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


def gen_input_signal(size, dt):
    signal = gen_filtered_white_noise(size, dt, 50)
    threshold = 0.002
    rectified = np.maximum(threshold, signal) - threshold
    normalized = 0.05 * rectified / rectified.max()
    return normalized + 0.0005


def gen_spike_train(input_signal, t_stop, dt):
    max_rate = input_signal.max() / dt

    def modulation(ts):
        return np.interp(
            ts.rescale(pq.ms).magnitude,
            np.arange(input_signal.size) * dt, input_signal) / input_signal.max()

    return stg.gen_inhomogeneous_poisson(
        modulation, max_rate / pq.ms, t_stop=t_stop * pq.ms,
        refractory=5 * pq.ms)


def get_spiketimes_list(trains):
    spikes = []
    for i, train in enumerate(trains):
        spikes.extend((i, spike.rescale(pq.s).magnitude * brian.second)
                      for spike in train)
    spikes.sort(key=lambda (i, t): t)
    return spikes


def gen_input_spiketimes(
        num_tunings, num_excitatory_per_tuning, num_inhibitory_per_tuning):
    t_stop = 500
    dt = 0.1
    size = int(t_stop / dt)
    excitatory = []
    inhibitory = []
    for i in xrange(num_tunings):
        input_signal = gen_input_signal(size, dt)
        excitatory.extend(gen_spike_train(input_signal, t_stop, dt)
                          for i in xrange(num_excitatory_per_tuning))
        inhibitory.extend(gen_spike_train(input_signal, t_stop, dt)
                          for i in xrange(num_inhibitory_per_tuning))
    return get_spiketimes_list(excitatory + inhibitory)
