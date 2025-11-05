import json
import sys
import os
import numpy as np
import time
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
from svzerodtrees import *
from pathlib import Path
from svzerodtrees.tune_bcs import *
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from scipy.optimize import minimize
from svzerodtrees.adaptation import *
from svzerodtrees.io import ConfigHandler
import pickle


def test_config_handler():
    '''
    test the config handler with a 3d-0d coupling file
    '''
    # load the config file
    threed_coupling_config = 'tests/case s/threed_cylinder/Simulations/threed_cylinder_rigid/svzerod_3Dcoupling.json'

    config_handler = ConfigHandler.from_json(threed_coupling_config, is_pulmonary=False, is_threed_interface=True)

    print(config_handler.config)


def test_coupled_tree_construction():
    '''
    test the construction of a coupled tree
    '''
    # load the config file
    threed_coupling_config = 'tests/cases/threed_cylinder/Simulations/threed_cylinder_rigid/svzerod_3Dcoupling.json'
    simulation_dir = 'tests/cases/threed_cylinder/Simulations/threed_cylinder_rigid/'

    config_handler = ConfigHandler.from_json(threed_coupling_config, is_pulmonary=False, is_threed_interface=True)

    preop.construct_coupled_cwss_trees(config_handler, simulation_dir)


def test_interface():
    '''
    test the interface
    '''
    preop_dir = '../threed_models/AS2_opt_fs/preop'
    postop_dir = '../threed_models/AS2_opt_fs/postop'
    adapted_dir = '../threed_models/AS2_opt_fs/adapted'
    zerod_config = '../threed_models/AS2_opt_fs/zerod/preop_config.json'
    interface.run_threed_from_msh(preop_dir, postop_dir, adapted_dir, zerod_config)


def test_steady_sim_setup():
    '''
    test the setup of a steady simulation
    '''
    # load the config file
    simulation_dir = 'cases/threed/SU0243/'

    simulation = SimulationDirectory.from_directory(simulation_dir, convert_to_cm=True)

    # simulation.generate_steady_sim()

    simulation.generate_simplified_zerod()


def test_sim_dir():
    '''
    test the simulation directory
    '''
    os.chdir('cases/threed/SU0243/')

    sim = Simulation(zerod_config='preop/SU0243_optimized.json', adapted_dir='adapted-cwss-ims', adaptation_config={
        "location": "uniform",
        "method": "wss-ims",
        "iterations": 100,
    })

    sim.run_pipeline(False, False, False)


def test_adapt_trees():


    jeff_dir = '/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/PPAS/tof-stent/TST-STAN-5/TST-STAN-5-pre-jeff-li/Simulations/cwss-adaptation'

    preop_sim_dir = SimulationDirectory.from_directory(jeff_dir + '/preop')
    postop_sim_dir = SimulationDirectory.from_directory(jeff_dir + '/postop')
    adapted_sim_dir = SimulationDirectory.from_directory(jeff_dir + '/adapted')

    clinical_targets = ClinicalTargets.from_csv('/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/PPAS/tof-stent/TST-STAN-5/clinical_targets.csv')

    adaptor = MicrovascularAdaptor(
        preop_sim_dir,
        postop_sim_dir,
        adapted_sim_dir,
        clinical_targets,
        bc_type="resistance"
    )


    t_start = time.time()
    n_iter = 1
    adaptor.adapt_resistance(n_iter=n_iter, d_min=0.05, coupler_path=f'adapted_{n_iter}iter.json', max_workers=1, parallel=False)

    t_end = time.time()

    print(f"Adaptation took {t_end - t_start} seconds")
    # non-parallel: ~55 seconds with d_mi=0.05
    # parallel: ~31 seconds

if __name__ == '__main__':
    # test_adapt_trees()

    # test_sim_dir()

    test_adapt_trees()
