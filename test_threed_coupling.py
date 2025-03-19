import json
import sys
import os
import numpy as np
import time
sys.path.append('/home/ndorn/Documents/Stanford/PhD/Simvascular/svZeroDPlus/structured_trees/src')
from svzerodtrees.structuredtree import StructuredTree
from pathlib import Path
from svzerodtrees.post_processing.stree_visualization import *
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from scipy.optimize import minimize
from svzerodtrees.adaptation import *
from svzerodtrees import operation, preop, interface
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees.result_handler import ResultHandler
from svzerodtrees.simulation_directory import *
from svzerodtrees.simulation import *
from svzerodtrees.threedutils import *
import pickle


def test_config_handler():
    '''
    test the config handler with a 3d-0d coupling file
    '''
    # load the config file
    threed_coupling_config = 'tests/cases/threed_cylinder/Simulations/threed_cylinder_rigid/svzerod_3Dcoupling.json'

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


def test_data_handling():
    '''
    test random data handlers'''

    Q_svZeroD = 'tests/cases/test_cylinder/Simulations/steady/Q_svZeroD'

    df = get_outlet_flow(Q_svZeroD)

    print(df)

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

    sim = Simulation(zerod_config='preop/SU0243_optimized.json')

    sim.run_pipeline(False, True)

def test_adapt_trees():

    preop_sim_dir = SimulationDirectory.from_directory('cases/threed/SU0243/preop')
    postop_sim_dir = SimulationDirectory.from_directory('cases/threed/SU0243/postop')
    adapted_sim_path = 'cases/threed/SU0243/adapted'

    adapt_threed(preop_sim_dir, postop_sim_dir, adapted_sim_path)




if __name__ == '__main__':
    # test_adapt_trees()

    test_sim_dir()
