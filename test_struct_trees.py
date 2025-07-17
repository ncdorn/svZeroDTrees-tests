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
from scipy.integrate import solve_ivp


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


def test_single_tree_adaptation():
    '''
    test the adaptation of a tree
    '''
    # [19992500.0, -30.70380829, 0.0, 41.41957157, 0.15439045]
    k1_l = 19992500
    k2_l = -30.70380829
    k3_l = 0.0
    lrr_l = 41.41957157
    d_l = 0.15439045
    d_min = 0.01

    time_array = np.linspace(0, 1, 512)

    test_tree = StructuredTree(name='test', time=time_array, simparams=None) 
    print(f'building test tree...')
    test_tree.build_tree(initial_d=d_l, d_min=d_min, lrr=lrr_l)

    print("number of vessels in the tree:", test_tree.count_vessels())

    with open(f'cases/zerod/tree-adaptation/tree_config_{d_l}_{d_min}.json', 'w') as f:
        json.dump(test_tree.block_dict, f, indent=4)

    test_tree2 = copy.deepcopy(test_tree)

    test_tree3 = copy.deepcopy(test_tree)

    R_initial = test_tree.root.R_eq
    
    print(f"preop tree resistance: {R_initial}")
    # cwss-ims adaptation
    # test_tree.adapt_wss_ims(Q=5.875246798749999, Q_new=6.577696098424999, n_iter=500)

    # cwss adaptation
    test_tree2.adapt_constant_wss(Q=5.875246798749999, Q_new=6.577696098424999, n_iter=1)

    # cwss-ims adaptation method 2
    print(f"adapting tree with cwss-ims method 2...")
    test_tree3.adapt_wss_ims_method2(Q=5.875246798749999, Q_new=6.577696098424999, n_iter=500)

    print(f"preop tree resistance: {R_initial}") # preop tree resistance: 238903.67921232135

    # print(f"cwss-ims adapted tree resistance: {test_tree.root.R_eq}") 
    # cwss-ims postop tree resistance: 229506.06467586374
        # with negative thickness gain: 228316.74880420387


    print(f"cwss-ims adapted tree resistance method 2: {test_tree3.root.R_eq}") 
    # cwss-ims adapted tree resistance method 2:
    # 500 iterations 213495.17043033676 for all parameters 0.00001
    # for all parameters 0.0001: 213388.98984401594 but we encounter a negative thickness

    # update the inflow for each iteration

    print(f"cwss adapted tree resistance: {test_tree2.root.R_eq}") # cwss adapted tree resistance: 213390.5330223265



    with open(f'cases/zerod/tree-adaptation/tree_config_{d_l}_{d_min}_adapted.json', 'w') as f:
        json.dump(test_tree.block_dict, f, indent=4)

def test_bifurcation_tree_adaptation():

    clinical_targets = ClinicalTargets.from_csv('cases/threed/SU0243/clinical_targets.csv')

    opt_params = pd.read_csv('cases/threed/SU0243/optimized_params.csv')
    tree_params = {
        'lpa': [opt_params['k1'][opt_params.pa=='lpa'].values[0], opt_params['k2'][opt_params.pa=='lpa'].values[0], opt_params['k3'][opt_params.pa=='lpa'].values[0], opt_params['lrr'][opt_params.pa=='lpa'].values[0], 0.9, 0.6],
        'rpa': [opt_params['k1'][opt_params.pa=='rpa'].values[0], opt_params['k2'][opt_params.pa=='rpa'].values[0], opt_params['k3'][opt_params.pa=='rpa'].values[0], opt_params['lrr'][opt_params.pa=='rpa'].values[0], 0.9, 0.6]
    }

    print(f"tree parameters: {tree_params}")

    d_l = opt_params['diameter'][opt_params.pa=='lpa'].values[0]
    d_r = opt_params['diameter'][opt_params.pa=='rpa'].values[0]

    print(f"lpa diameter: {d_l}, rpa diameter: {d_r}")

    # flow split right: 0.7998511032283667,

    os.system('pwd')

    preop_pa_config_handler = ConfigHandler.from_json('cases/zerod/tree-adaptation/simple_pa/preop_pa_config.json', is_pulmonary=True)
    postop_pa_config_handler = ConfigHandler.from_json('cases/zerod/tree-adaptation/simple_pa/postop_pa_config.json', is_pulmonary=True)

    preop_pa_config = PAConfig.from_pa_config(preop_pa_config_handler, clinical_targets)
    postop_pa_config = PAConfig.from_pa_config(postop_pa_config_handler, clinical_targets)

    preop_pa_config.create_steady_trees(d_l, d_r, [0.5, 0.5], tree_params, 24)
    postop_pa_config.create_steady_trees(d_l, d_r, [0.5, 0.5], tree_params, 24)

    preop_pa_config.simulate()
    postop_pa_config.simulate()
    print(f"preop rpa split: {preop_pa_config.rpa_split}")
    print(f"postop rpa split: {postop_pa_config.rpa_split}")

    preop_lpa_flow = np.mean(preop_pa_config.result[preop_pa_config.result.name=='branch2_seg0']['flow_out'])
    postop_lpa_flow = np.mean(postop_pa_config.result[postop_pa_config.result.name=='branch2_seg0']['flow_out'])
    preop_rpa_flow = np.mean(preop_pa_config.result[preop_pa_config.result.name=='branch4_seg0']['flow_out'])
    postop_rpa_flow = np.mean(postop_pa_config.result[postop_pa_config.result.name=='branch4_seg0']['flow_out'])


    def adapt_cwss(pa_config, n_iter=100):
        '''
        adapt the cwss of the pa config
        '''
        print(f"computing cwss adaptation for {n_iter} iterations...")
        cwss_pa_config = copy.deepcopy(pa_config)
        for i in range(n_iter):
            print(f"iteration {i+1} of {n_iter}")
            cwss_pa_config.lpa_tree.adapt_constant_wss(Q=preop_lpa_flow, Q_new=postop_lpa_flow, n_iter=1) # there is some difference between this and the tree in single vessel adaptation and it is taking much longet
            cwss_pa_config.rpa_tree.adapt_constant_wss(Q=preop_rpa_flow, Q_new=postop_rpa_flow, n_iter=1)
        cwss_pa_config.update_bcs()
        cwss_pa_config.simulate()

        print(f"rpa split after {n_iter} iteration cwss adaptation: {cwss_pa_config.rpa_split}")

    adapt_cwss(postop_pa_config, n_iter=1)




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
    # test_single_tree_adaptation()5
    test_bifurcation_tree_adaptation()


