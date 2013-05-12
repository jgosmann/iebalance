#!/usr/bin/env python

import config
import re


prefix_to_latex = {
    'm': r'\milli',
    'n': r'\nano',
    'p': r'\pico'
}

unit_to_latex = {
    'amp': r'\ampere',
    'hertz': r'\hertz',
    'second': r'\second',
    'siemens': r'\siemens',
    'volt': r'\volt'
}


# From http://www.peterbe.com/plog/uniqifiers-benchmark
def unique(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x):
            return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def beta_extract(equations):
    look_for = re.compile(r'^\s*w\s*\+=.*-[^-+]*(\d+\.\d+).*$')
    for eqn in equations:
        m = look_for.match(eqn)
        if m is not None:
            return m.group(1)
    return None


def quantity((value, unit)):
    try:
        latex_unit = unit_to_latex[unit]
    except:
        latex_unit = prefix_to_latex[unit[0]] + unit_to_latex[unit[1:]]
    return r'\SI{%s}{%s}' % (str(value), latex_unit)


def start_section(title):
    print r'''
        \begin{tabularx}{\linewidth}{@{}llX@{}} \\
            \multicolumn{3}{c}{\cellcolor{black!33}\textsf{%s}} \\ \addlinespace
        Name & Value(s) & Description \\
        \cmidrule(r){1-1} \cmidrule(lr){2-2} \cmidrule(l){3-3} \addlinespace
        ''' % title


def end_section():
    print r'''
            \addlinespace \bottomrule
        \end{tabularx}
        '''


class ParamWriter(object):
    def __init__(self, filenames):
        self.filenames = filenames
        try:
            files = [tables.openFile(filename, 'r') for filename in filenames]
            self.configs = [f.getNodeAttr('/', 'config') for f in files]
        finally:
            for f in files:
                f.close()

    @staticmethod
    def get_conf_value(config, type, path):
        value = config
        for key in path:
            value = value[key]
        return type(value)

    def get_conf_values(self, type, path):
        return unique(self.get_conf_value(c, type, path) for c in self.configs)

    def write(self, type, path, name, description, from_file=None):
        if from_file is None:
            values = map(str, self.get_conf_values(type, path))
        else:
            idx = self.filenames.index(from_file)
            values = unique([str(
                self.get_conf_value(self.configs[idx], type, path))])
        print r'$%s$ & %s & %s \\' % (name, ", ".join(values), description)


if __name__ == "__main__":
    import numpy as np
    import tables

    filenames = [
        'Data/simdata-long.h5',
        'Data/simdata-nodecay.h5',
        'Data/simdata-long-t120s.h5',
        'Data/simdata-weight-dependent.h5',
        'Data/trained-long-filter.h5',
        'Data/exc-stdp-mult-l2-r1.h5'
    ]
    p = ParamWriter(filenames)

    start_section('Input')
    p.write(
        int, ['inputs', 'num_tunings'], r'N_S',
        'Number of different rate input signals')
    num_trains = np.array([c['inputs']['num_trains'] for c in p.configs])
    inh_fractions = np.array(
        [c['inputs']['fraction_inhibitory'] for c in p.configs])
    num_exc = set(num_trains * (1 - inh_fractions))
    print r'$N_{\text{E}}$ & %s & Number of excitatory input spike trains\\' %\
        ", ".join(str(int(n)) for n in num_exc)
    num_inh = set(num_trains * inh_fractions)
    print r'$N_{\text{I}}$ & %s & Number of inhibitory input spike trains\\' %\
        ", ".join(str(int(n)) for n in num_inh)
    p.write(
        quantity, ['inputs', 'refractory_period'], r'\tau_{\text{ref},\text{in}}',
        'Absolute refractory period')
    p.write(
        quantity, ['inputs', 'raw_signals', 'filter_time_constant'],
        r'\tau_s', 'Autocorrelation time constant of rate modulation')
    end_section()

    start_section('Neuron Model')
    p.write(quantity, ['model', 'tau'], r'\tau', 'Membrane time constant')
    p.write(quantity, ['model', 'threshold'], r'\Theta', 'Spiking threshold')
    p.write(
        quantity, ['model', 'V_rest'], r'V^{\text{rest}}', 'Resting potential')
    p.write(
        quantity, ['model', 'V_exc'], r'V^{\text{E}}',
        'Excitatory reversal potential')
    p.write(
        quantity, ['model', 'V_inh'], r'V^{\text{I}}',
        'Inhibitory reversal potential')
    p.write(
        quantity, ['model', 'g_leak'], r'g^{\text{leak}}',
        'Leak conductance')
    p.write(
        quantity, ['model', 'refractory_period'], r'\tau_{\text{ref}}',
        'Absolute refractory period')
    end_section()

    start_section('Synapse Model')
    p.write(
        quantity, ['model', 'tau_exc'], r'\tau_{\text{E}}',
        'Decay constant of excitatory synaptic conductance')
    p.write(
        quantity, ['model', 'tau_inh'], r'\tau_{\text{I}}',
        'Decay constant of inhibitory synaptic conductance')
    end_section()

    start_section('Plasticity Model')
    p.write(
        quantity, ['model', 'tau_stdp'], r'\tau_{\text{STDP}}',
        'Decay constant of (pre and post) synaptic trace')
    p.write(float, ['model', 'eta'], r'\eta\I{}', 'Learning rate of inhibitory synapses')
    print r'$\eta\E{}$ & 0.0025, 0.005, 0.0075 & Learning rate of excitatory synapses\\'
    init_inh_w = set(str(
        config.quantity(c['model']['init_inh_w']) /
        config.quantity(c['model']['g_inh_bar'])) for c in p.configs)
    print r'$w\I{,i}(\SI{0}{\second})$ & %s & Initial inhibitory synaptic weights\\' % \
        ", ".join(init_inh_w)
    init_exc_w = r''''''
    print r'''$w\E{,j}(\SI{0}{\second})$ &
        $0.3 + \frac{1.1}{1 + (K(j) - P)^4} + \xi$ &
        Initial excitatory synaptic weights with input signal number
        $K(i) \in [1, N_S]$ of synapse $j$, position of the tuning curve peak $P$,
        and a noise term $\xi \in [0, 0.1]$\\'''
    print r'$P$ & 5 & Peak of the tuning curve\\'
    p.write(
        quantity, ['model', 'g_exc_bar'], r'\bar g_{\text{E}}',
        'Excitatory synaptic weight change to conductance conversion factor')
    p.write(
        quantity, ['model', 'g_inh_bar'], r'\bar g_{\text{I}}',
        'Inhibitory synaptic weight change to conductance conversion factor')
    p.write(
        quantity, ['model', 'rho'], r'\rho_0',
        r'Target firing rate for the \ref{eqn:lr-orig} learning rule',
        filenames[0])
    p.write(
        quantity, ['model', 'tau_w'], r'\tau_w',
        r'Decay constant of inhibitory synaptic weights for the \ref{eqn:lr-exp-decay} learning rule',
        filenames[2])
    p.write(
        beta_extract, ['model', 'synapses', 'inhibitory', 'pre'], r'\beta',
        r'Depression factor for the \ref{eqn:lr-weight-dep} learning rule',
        filenames[3])
    end_section()
