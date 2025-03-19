import svzerodtrees
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees.inflow import Inflow
from svzerodtrees.structuredtree import *
from svzerodtrees.preop import *
import pysvzerod
import pickle
import os
import numpy as np


def build_tree(d_root, k1, k2, lrr, config_handler):
        
        inflow = Inflow.periodic()
        inflow.rescale(tsteps=500, cardiac_output=1.0)

        config_handler.set_inflow(inflow)
        tree = StructuredTree(name='test_impedance', time=config_handler.bcs['INFLOW'].values['t'], Pd=6.0, simparams=config_handler.simparams)
        tree.build_tree(d_root, 0.01, alpha=0.9, beta=0.6, xi=2.7, lrr=lrr)

        tree.compute_olufsen_impedance(k1=k1, k2 = k2, k3 = 0.0, n_procs=24)

        config_handler.bcs['RCR_0'] = tree.create_impedance_bc('RCR_0', 6.0 * 1333.2)

        # get result and return pressure in
        try:
            result = pysvzerod.simulate(config_handler.config)
            t = result[result.name=='branch0_seg0']['time']
            # normalized pressure
            # p = result[result.name=='branch0_seg0']['pressure_in'] / 1333.2 / max(result[result.name=='branch0_seg0']['pressure_in'] / 1333.2)
            # unnormalized pressure
            p = result[result.name=='branch0_seg0']['pressure_in'] / 1333.2
        except:
            t = 0.0
            p = 0.0

        return t, p

def compare_stiffness():
    '''
    compare the stiffness values of the structured tree and the effect on mean pressure and waveform shape
    '''

    # create a simple comfig
    config_handler = ConfigHandler.from_json('cases/zerod/simple_config/simple_config_1rcr.json', is_pulmonary=False)

    # assemble parameters   
    d_roots = [0.2]

    k1_val = 19992500,
    k2 = [-75, -90, -100]

    lrr = [10, 25, 50]

    # create subplots
    fig, ax = plt.subplots(1, 1)

    for i, d_root in enumerate(d_roots):
        for j, k2_val in enumerate(k2):
            for l, lrr_val in enumerate(lrr):
                print(f'k1: {k1_val}, k2: {k2_val}, lrr: {lrr_val}')
                t, p = build_tree(d_root, k1_val, k2_val, lrr_val, config_handler)

                ax.plot(t, p, label=f'k1={k1_val}, k2={k2_val}, lrr={lrr_val}')
                ax.legend()

    plt.show()

    config_handler.to_json('cases/zerod/impedance/pa_tuning/tree_param_test.json')


def run_test_config():
    '''run the test config
    '''

    config_handler = ConfigHandler.from_json('cases/zerod/impedance/pa_tuning/tree_param_test.json', is_pulmonary=False)

    result = pysvzerod.simulate(config_handler.config)

    print('simulation complete!')

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(result[result.name=='branch0_seg0']['time'], result[result.name=='branch0_seg0']['pressure_in'] / 1333.2, label='pressure in')

    ax[1].plot(result[result.name=='branch0_seg0']['time'], result[result.name=='branch0_seg0']['flow_in'] / 1333.2, label='flow in')

if __name__ == '__main__':

    compare_stiffness()

