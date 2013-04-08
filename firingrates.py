#!/usr/bin/env python

import logging

import brian as b
from joblib import Parallel, delayed
import tables

from config import Configurable, quantity, quantity_list
import singlecell


logger = logging.getLogger('firing-rates')


class ModelRateRecorder(Configurable):
    def __init__(self, config, model):
        Configurable.__init__(self, config)
        self._add_config_value('store_times', quantity_list)
        self.model = model

        self.m_spike_count = b.PopulationSpikeCounter(model.neuron)

    def record(self):
        time_passed = 0 * b.second
        for i, time in enumerate(self.store_times):
            logger.info(
                'Running time interval %i (duration %ds, end time %ds)',
                i, time - time_passed, time)
            self.model.run(time - time_passed, report='text')
            self.model.add(self.m_spike_count)
            time_passed = time

        return float(
            self.m_spike_count.nspikes /
            (time_passed - self.store_times[0]) / b.hertz)


def run_simulation(config, tau_w):
    config['model']['tau_w'] = [tau_w / b.second, "second"]
    #b.Clock(quantity(config['dt']), makedefaultclock=True)
    model = singlecell.SingleCellModel(config)
    recorder = ModelRateRecorder(config['recording'], model)
    return [tau_w, recorder.record()]


class RatesTable(tables.IsDescription):
    tau_w = tables.Float32Col()
    rate = tables.Float32Col()


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
        description="Run the Vogels et al. 2011 single cell model and record " +
        "firing rates for different time scale paramaters tau_d.")
    parser.add_argument(
        '-c', '--config', type=str, nargs=1, required=True,
        help="Path to the configuration file.")
    parser.add_argument(
        '-j', '--jobs', type=int, nargs=1, default=[1],
        help="Number of parallel jobs.")
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

    tau_ws = quantity_list(config['tau_ws'])

    b.defaultclock.dt = quantity(config['dt'])
    data = Parallel(n_jobs=args.jobs[0])(delayed(run_simulation)(config, tau_w)
                                         for tau_w in tau_ws)
    print data

    with tables.openFile(os.path.join(outpath, args.output[0]), 'w') as outfile:
        outfile.setNodeAttr('/', 'config', config)
        table = outfile.createTable('/', 'rates', RatesTable)
        table.attrs.unit = 'hertz'
        table.append(data)
        outfile.flush()
