import json
import sys
import os
import numpy as np
import pandas as pd
import scipy.signal
# print(sys.path)
import svzerodtrees
import svzerodtrees.inflow
from svzerodtrees.structuredtree import StructuredTree
from pathlib import Path
from svzerodtrees.post_processing.stree_visualization import *
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from scipy.optimize import minimize
from svzerodtrees.adaptation import *
from svzerodtrees.preop import *
from svzerodtrees.structuredtree import StructuredTree
from svzerodtrees.config_handler import ConfigHandler, SimParams
from svzerodtrees.result_handler import ResultHandler
import pickle
import scipy
import pysvzerod


def build_simple_tree():
    '''
    build a simple tree from a config for testing
    '''
    
    os.chdir('tests/cases/simple_config')
    input_file = 'simple_config_1out.json'
    
    config_handler = ConfigHandler.from_json(input_file)

    result_handler = ResultHandler.from_config_handler(config_handler)

    

def build_tree_R_optimized():
    '''
    build a tree from the class method
    '''

    tree = StructuredTree(name='test_tree')
    

    tree.optimize_tree_diameter(resistance=100.0)

    # example: compute pressure and flow in the tree with inlet flow 10.0 cm3/s and distal pressure 100.0 dyn/cm2
    tree_result = tree.simulate(Q_in = [10.0, 10.0], Pd=100.0)

    # example: adapt the tree
    R_old, R_new = tree.adapt_constant_wss(10.0, 5.0)


    print(f'R_old = {R_old}, R_new = {R_new}')


def test_fft():
    '''
    test the olufsen imedance calculation
    '''
    # test fft
    with open('tests/cases/pa_unsteady/inflow.flow') as ff:
        inflow = pd.read_csv(ff, delimiter=' ', header=None, names=['t', 'q'])
    
    inflow['q'] = inflow['q'] * -1
    
    Y = np.fft.fft(inflow['q'])

    Y_half = copy.deepcopy(Y)

    np.put(Y_half, range(101, 201), 0.0)

    print(Y_half, Y)

    y_half = np.fft.ifft(Y_half)
    y = np.fft.ifft(Y)


    plt.plot(inflow.t, inflow.q, label='original signal')
    plt.plot(inflow.t, y_half, label='first n/2 fft components')
    plt.plot(inflow.t, y, '--', label='full fft components')
    plt.legend()
    plt.show()


def test_impedance_trees():
    '''
    test the impedance calculations in the frequence domain
    
    it is interesting that the flow and pressure does not actually depend on the outlet flow or pressure.
    we just sample some frequencies in the time period of the inflow and calculate the impedance at each frequency'''

    # enter simulation directory
    os.chdir('cases/threed/LPA_RPA')

    # config_handler = ConfigHandler.from_json('zerod_config.json')

    # clinical_targets = ClinicalTargets.from_csv('clinical_targets.csv')

    # construct_impedance_trees(config_handler, 'mesh-complete/mesh-surfaces', clinical_targets, d_min=0.05)

    # config_handler.to_json('zerod_config_impedance.json')

    # result = pysvzerod.simulate(config_handler.config)

    with open('zerod_config_impedance.json') as f:
        config = json.load(f)

    result = pysvzerod.simulate(config)

    print('simulation complete!')

    time = result[result['name'] == 'branch0_seg0']['time'].values

    mpa_flow = result[result['name'] == 'branch0_seg0']['flow_in'].values
    lpa_flow = result[result['name'] == 'branch1_seg0']['flow_in'].values
    rpa_flow = result[result['name'] == 'branch2_seg0']['flow_in'].values

    mpa_pressure = result[result['name'] == 'branch0_seg0']['pressure_in'].values / 1333.2
    lpa_pressure = result[result['name'] == 'branch1_seg2']['pressure_out'].values / 1333.2
    rpa_pressure = result[result['name'] == 'branch2_seg2']['pressure_out'].values / 1333.2

    # plot the pressures and flows
    fig, axs = plt.subplots(3, 1)

    # pressure figure
    axs[0].plot(time, mpa_pressure, label='MPA pressure')
    axs[0].plot(time, lpa_pressure, label='LPA pressure')
    axs[0].plot(time, rpa_pressure, label='RPA pressure')
    axs[0].set_xlabel('time [s]')
    axs[0].set_ylabel('pressure [dyn/cm^2]')
    axs[0].legend()

    # flow figure
    axs[1].plot(time, mpa_flow, label='MPA flow')
    axs[1].plot(time, lpa_flow, label='LPA flow')
    axs[1].plot(time, rpa_flow, label='RPA flow')
    axs[1].set_xlabel('time [s]')
    axs[1].set_ylabel('flow [cm^3/s]')
    axs[1].legend()

    # pressure-flow figure
    axs[2].plot(mpa_flow[100:], mpa_pressure[100:], label='MPA')
    axs[2].plot(lpa_flow[100:], lpa_pressure[100:], label='LPA')
    axs[2].plot(rpa_flow[100:], rpa_pressure[100:], label='RPA')
    axs[2].set_xlabel('flow [cm^3/s]')
    axs[2].set_ylabel('pressure [dyn/cm^2]')
    axs[2].legend()


    plt.tight_layout()
    plt.show()


def test_tree_adaptation():
    '''
    test the adaptation of a tree
    '''

    k1_l = 19992500
    k2_l = -25
    k3_l = 0.0
    lrr_l = 10.0
    d_l = 0.2
    d_min = 0.2

    time_array = np.linspace(0, 1, 512)

    test_tree = StructuredTree(name='test', time=time_array, simparams=None) 
    print(f'building test tree...')
    test_tree.build_tree(initial_d=d_l, d_min=d_min, lrr=lrr_l)

    with open(f'cases/zerod/tree-adaptation/tree_config_{d_l}_{d_min}.json', 'w') as f:
        json.dump(test_tree.block_dict, f, indent=4)

    print(f"preop tree resistance: {test_tree.root.R_eq}")

    test_tree.adapt_wss_ims(Q=10.0, Q_new=20.0)

    print(f"postop tree resistance: {test_tree.root.R_eq}")


    with open(f'cases/zerod/tree-adaptation/tree_config_{d_l}_{d_min}_adapted.json', 'w') as f:
        json.dump(test_tree.block_dict, f, indent=4)



def fix_zerod_config():
    '''
    fix the zerod config for the impedance trees
    '''

    os.chdir('cases/threed/LPA_RPA')

    inflow = svzerodtrees.inflow.Inflow.periodic(path='inflow.flow', t_per = 1.0, flip_sign=True)

    inflow.rescale(tsteps=1024, t_per=1.0)


    config_handler = ConfigHandler.from_json('zerod_config.json')

    config_handler.set_inflow(inflow)

    config_handler.to_json('zerod_config.json')

if __name__ == '__main__':

    # fix_zerod_config()
    test_tree_adaptation()


