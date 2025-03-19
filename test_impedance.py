import svzerodtrees
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees.inflow import Inflow
from svzerodtrees.structuredtree import StructuredTree
from svzerodtrees.utils import *
from svzerodtrees.simulation_directory import *
from svzerodtrees.preop import *
import pysvzerod
import pickle
import os
import numpy as np
from scipy.optimize import minimize, Bounds

def test_impedance_tree(build_tree=True):
    '''
    test the impedance tuning
    '''

    os.chdir('cases/zerod/impedance/tuning')
    # load the config
    if build_tree:
        config_handler = ConfigHandler.from_json('one_tree.json', is_pulmonary=False)
    else:
        config_handler = ConfigHandler.from_json('one_tree_imp_dmin01.json', is_pulmonary=False)

    # get flow from pulmonary result
    simulation = SimulationDirectory.from_directory('../../../../../threed_models/SU0243/preop', convert_to_cm=True)
    time, flow, pressure = simulation.svzerod_data.get_result(simulation.svzerod_3Dcoupling.coupling_blocks['RESISTANCE_0'])
    cap_d = (simulation.mesh_complete.mesh_surfaces[1].area / np.pi)**(1/2) * 2 / 10

    inflow = svzerodtrees.inflow.Inflow(q=flow[-500:], t=time[:500], t_per=1.0, n_periods=1, name='INFLOW')
    inflow.rescale(tsteps=512, t_per=1.0)

    config_handler.set_inflow(inflow)

    if build_tree:

        tree = StructuredTree(name='test_impedance', time=config_handler.bcs['INFLOW'].values['t'], Pd=6.0, simparams=config_handler.simparams)

        # from optimziation with d_min=0.01: [k2: -36.26842638750388, k3: 0.000742529872020475, alpha: 0.9, lrr: 66.4432369501307]
        tree.build_tree(cap_d, 0.01, alpha=0.9, beta=0.6, xi=2.7, lrr=66.5)

        tree.compute_olufsen_impedance(k2 = -36, k3 = 0.0, n_procs=24)

        config_handler.bcs['RESISTANCE_0'] = tree.create_impedance_bc('RESISTANCE_0', 6.0 * 1333.2)

        config_handler.to_json('one_tree_imp_dmin01.json')

    result = pysvzerod.simulate(config_handler.config)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    t = result[result.name=='branch1_seg0']['time']
    p = result[result.name=='branch1_seg0']['pressure_out'] / 1333.2
    q = result[result.name=='branch1_seg0']['flow_in']
    # flow plot
    axs[0].plot(t, q, label='0D')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Flow (mL/s)')
    # axs[0].plot(time, flow, label='3D')
    axs[1].plot(t, p)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Pressure (mmHg)')
    # add mean pressure as a horizontal line
    axs[1].axhline(y=np.mean(p), color='r', linestyle='--', label='mean pressure')
    axs[1].legend()
    # axs[1].plot(time, pressure / 1333.2, label='3D')
    axs[2].plot(q, p, label='0D')
    axs[2].set_xlabel('Flow (mL/s)')
    axs[2].set_ylabel('Pressure (mmHg)')
    # axs[2].plot(flow, pressure / 1333.2, label='3D')

    plt.legend()
    plt.show()


    print(result)

def tune_tree():
    '''
    tune the tree parameters to some systolic, diastolic and mean pressure
    '''

    os.chdir('cases/zerod/impedance/tuning')

    config_handler = ConfigHandler.from_json('one_tree.json', is_pulmonary=False)

    
    def optimize_tree(params, pressures, config_handler, cap_d):
        '''
        optimize the tre
        :param params: [k2, k3, alpha, lrr]
        :pram pressures: [sys, dias, mean]'''

        config_handler_imp = build_tree(params[0], params[1], 0.9, params[2], config_handler, cap_d=cap_d)

        print(f'Optimizing with parameters: k2: {params[0]}, k3: {params[1]}, alpha: 0.9, lrr: {params[2]}')

        try:
            result = pysvzerod.simulate(config_handler_imp.config)

            p = result[result.name=='branch1_seg0']['pressure_out'] / 1333.2

            sys_p = np.max(p[-512:])
            dias_p = np.min(p[-512:])
            mean_p = np.mean(p[-512:])

            loss = (sys_p - pressures[0])**2 + (dias_p - pressures[1])**2 + (mean_p - pressures[2])**2

        except:
            loss = 1e6
            sys_p = 0
            dias_p = 0
            mean_p = 0

        print(f'loss: {loss}, sys: {sys_p}, dias: {dias_p}, mean: {mean_p}')

        return loss
    

    pressures = [72, 14, 38]

    cap_d = 0.3213

    # with alpha
    # bounds = Bounds(lb=[-np.inf, 0.0, 0.0, 0.0], ub= [np.inf, np.inf, 1.0, np.inf])

    # no alpha
    bounds = Bounds(lb=[-np.inf, 0.0, 0.0], ub= [np.inf, np.inf, np.inf])

    res = minimize(optimize_tree, [-34.05935937, 0.0, 98.81390018], args=(pressures, config_handler, cap_d), method='Nelder-Mead', bounds=bounds)

    print(f'Optimized parameters: {res.x}')

    optimized_config = build_tree(res.x[0], res.x[1], 0.9, res.x[2], config_handler, cap_d)

    optimized_config.to_json('optimized_imp.json')

    result = pysvzerod.simulate(optimized_config.config)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    t = result[result.name=='branch1_seg0']['time']
    p = result[result.name=='branch1_seg0']['pressure_out'] / 1333.2
    q = result[result.name=='branch1_seg0']['flow_in']

    # flow plot
    axs[0].plot(t, q, label='0D')
    
    # pressure
    axs[1].plot(t, p, label='0D')

    # flow pressure
    axs[2].plot(q, p, label='0D')

    plt.show()


def run_optimized_tree():
    ''' simulate the optimized config '''
 
    os.chdir('cases/zerod/impedance/pa_tuning/nonlin/t2000')

    optimized_config = ConfigHandler.from_json('pa_config_onetree.json', is_pulmonary=False)

    result = pysvzerod.simulate(optimized_config.config)

    fig, axs = plt.subplots(3, 3, figsize=(10, 5))
    mpa_t = result[result.name=='branch0_seg0']['time']
    mpa_p = result[result.name=='branch0_seg0']['pressure_out'] / 1333.2
    mpa_q = result[result.name=='branch0_seg0']['flow_in']

    # mpa flow plot
    axs[0, 0].plot(mpa_t, mpa_q, label='0D')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Flow (mL/s)')
    axs[0, 0].set_title('Flow')

    # mpa pressure
    axs[1, 0].plot(mpa_t, mpa_p, label='0D')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Pressure (mmHg)')
    axs[1, 0].set_title('Pressure')

    # add horizontal line for mean pressure
    axs[1, 0].axhline(y=np.mean(mpa_p), color='r', linestyle='--', label='mean flow')

    # mpa flow-pressure
    axs[2, 0].plot(mpa_q, mpa_p, label='0D')
    axs[2, 0].set_xlabel('Flow (mL/s)')
    axs[2, 0].set_ylabel('Pressure (mmHg)')
    axs[2, 0].set_title('Flow-Pressure')

    # lpa
    lpa_t = result[result.name=='branch2_seg0']['time']
    lpa_p = result[result.name=='branch2_seg0']['pressure_out'] / 1333.2
    lpa_q = result[result.name=='branch2_seg0']['flow_out']

    # lpa flow plot
    axs[0, 1].plot(lpa_t, lpa_q, label='0D')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Flow (mL/s)')
    axs[0, 1].set_title('Flow')

    # lpa pressure
    axs[1, 1].plot(lpa_t, lpa_p, label='0D')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Pressure (mmHg)')
    axs[1, 1].set_title('Pressure')

    # add horizontal line for mean pressure
    axs[1, 1].axhline(y=np.mean(lpa_p), color='r', linestyle='--', label='mean flow')

    # lpa flow-pressure
    axs[2, 1].plot(lpa_q, lpa_p, label='0D')
    axs[2, 1].set_xlabel('Flow (mL/s)')
    axs[2, 1].set_ylabel('Pressure (mmHg)')
    axs[2, 1].set_title('Flow-Pressure')

    # rpa
    rpa_t = result[result.name=='branch4_seg0']['time']
    rpa_p = result[result.name=='branch4_seg0']['pressure_out'] / 1333.2
    rpa_q = result[result.name=='branch4_seg0']['flow_out']

    # rpa flow plot
    axs[0, 2].plot(rpa_t, rpa_q, label='0D')
    axs[0, 2].set_xlabel('Time (s)')
    axs[0, 2].set_ylabel('Flow (mL/s)')
    axs[0, 2].set_title('Flow')

    # rpa pressure
    axs[1, 2].plot(rpa_t, rpa_p, label='0D')
    axs[1, 2].set_xlabel('Time (s)')
    axs[1, 2].set_ylabel('Pressure (mmHg)')
    axs[1, 2].set_title('Pressure')

    # add horizontal line for mean pressure
    axs[1, 2].axhline(y=np.mean(rpa_p), color='r', linestyle='--', label='mean flow')

    # rpa flow-pressure
    axs[2, 2].plot(rpa_q, rpa_p, label='0D')
    axs[2, 2].set_xlabel('Flow (mL/s)')
    axs[2, 2].set_ylabel('Pressure (mmHg)')
    axs[2, 2].set_title('Flow-Pressure')


    plt.tight_layout()
    plt.show()


def build_tree(k2, k3, alpha, lrr, config_handler, cap_d=0.3213):
        


        tree = StructuredTree(name='test_impedance', time=config_handler.bcs['INFLOW'].values['t'], Pd=6.0, simparams=config_handler.simparams)


        tree.build_tree(cap_d, 0.01, alpha=alpha, beta=0.6, xi=2.7, lrr=lrr)

        tree.compute_olufsen_impedance(k2 = k2, k3 = k3, n_procs=24)

        config_handler.bcs['RESISTANCE_0'] = tree.create_impedance_bc('RESISTANCE_0', 6.0 * 1333.2)

        return config_handler


def test_tune_pa_trees():

    os.chdir('cases/zerod/impedance/pa_tuning/nonlin/t2000')

    # zerod_config_path = '/Users/ndorn/ndorn@stanford.edu - Google Drive/My Drive/Stanford/PhD/Simvascular/zerod_models/SU0243_prestent/SU0243_dmin01_lrr50.json'

    zerod_config_path = 'simplified_nonlin_zerod.json'
    msh_surf_path = '../../../../../../../threed_models/SU0243/preop/mesh-complete/mesh-surfaces'

    config_handler = ConfigHandler.from_json(zerod_config_path, is_pulmonary=True)

    clinical_targets = ClinicalTargets(wedge_p=5.0, mpa_p=[72.0, 14.0, 38.0], rpa_split=0.8)

    optimize_impedance_bcs(config_handler, msh_surf_path, clinical_targets, d_min=0.01, convert_to_cm=True, n_procs=24)


def test_construct_impedance_trees():
    
    os.chdir('cases/threed/SU0243/preop/')

    zerod_config_path = 'SU0243_optimized_fs8020.json'

    msh_surf_path = 'mesh-complete/mesh-surfaces'

    config_handler = ConfigHandler.from_json(zerod_config_path, is_pulmonary=False)

    # create an inflow with 1000 timesteps
    inflow = Inflow.periodic()
    inflow.rescale(tsteps=2000, cardiac_output=41.7)

    config_handler.set_inflow(inflow)

    clinical_targets = ClinicalTargets(wedge_p=5.0, mpa_p=[72.0, 14.0, 38.0], rpa_split=0.8)

    construct_impedance_trees(config_handler, msh_surf_path, clinical_targets.wedge_p, d_min=0.01, convert_to_cm=True, is_pulmonary=True, use_mean=True, specify_diameter=True, tree_params={'lpa': [19992500, -30.70380829, 0.0, 41.41957157, 0.15439045], 
                                                                                                                                                                                             'rpa': [19992500, -58.61992902, 0.0, 41.41957157, 0.25603551]}, 
                                                                                                                                                                                             n_procs=24)
    
    config_handler.to_json('SU0243_optimized.json')
    # config_handler.to_json('svzerod_3Dcoupling.json')


def compare_3d_0d_outlet():
    '''
    compare the outlet result of the 3D simulation to a 0D simulation
    '''

    sim = SimulationDirectory.from_directory('cases/threed/SU0243', convert_to_cm=True)

    time, flow3d, pres3d = sim.svzerod_data.get_result(sim.svzerod_3Dcoupling.coupling_blocks['RESISTANCE_0'])

    bc = sim.svzerod_3Dcoupling.bcs['RESISTANCE_0']

    inflow_0d = Inflow(q=flow3d[1500:2000], t=np.linspace(0, 1.0, 500), t_per=1.0, n_periods=1, name='INFLOW')

    inflow_0d.rescale(tsteps=512, t_per=1.0)

    zerod_config = ConfigHandler.from_json('cases/zerod/impedance/tuning/one_tree.json', is_pulmonary=False)

    zerod_config.set_inflow(inflow_0d)

    zerod_config.bcs['RESISTANCE_0'] = bc

    zerod_result = pysvzerod.simulate(zerod_config.config)

    flow0d = zerod_result[zerod_result.name=='branch1_seg0']['flow_out']
    pres0d = zerod_result[zerod_result.name=='branch1_seg0']['pressure_out'] / 1333.2

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].plot(time, flow3d, label='3D')
    axs[0].plot(zerod_result[zerod_result.name=='branch1_seg0']['time'], flow0d, label='0D')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Flow (mL/s)')
    axs[0].legend()

    axs[1].plot(time, pres3d / 1333.2, label='3D')
    axs[1].plot(zerod_result[zerod_result.name=='branch1_seg0']['time'], pres0d, label='0D')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Pressure (mmHg)')
    axs[1].legend()

    axs[2].plot(flow3d, pres3d, label='3D')
    axs[2].plot(flow0d, pres0d, label='0D')
    axs[2].set_xlabel('Flow (mL/s)')
    axs[2].set_ylabel('Pressure (mmHg)')
    axs[2].legend()

    plt.tight_layout()

    plt.show()


def test_rcr_comparison():
    '''
    Optimize RCR parameters against an optimized impedance tree for a PA config and compare the results
    '''

    os.chdir('cases/zerod/impedance/pa_tuning/nonlin')
    zerod_config_path = 'simplified_nonlin_zerod.json'

    msh_surf_path = '../../../../../../threed_models/SU0243/preop/mesh-complete/mesh-surfaces'

    config_handler = ConfigHandler.from_json(zerod_config_path, is_pulmonary=True)

    clinical_targets = ClinicalTargets(wedge_p=5.0, mpa_p=[72.0, 14.0, 38.0], rpa_split=0.8)

    # get the rpa/lpa info
    rpa_info, lpa_info, inflow_info = vtp_info(msh_surf_path, convert_to_cm=True, pulmonary=True)
    
    # rpa_mean_dia = 0.32
    rpa_mean_dia = np.mean([(area / np.pi)**(1/2) * 2 for area in rpa_info.values()])
    print(f'RPA mean diameter: {rpa_mean_dia}')
    # lpa_mean_dia = 0.32
    lpa_mean_dia = np.mean([(area / np.pi)**(1/2) * 2 for area in lpa_info.values()])
    print(f'LPA mean diameter: {lpa_mean_dia}')
    
    pa_config = PAConfig.from_pa_config(config_handler, clinical_targets)

    # rescale inflow by number of outlets ## TODO: figure out scaling for this
    pa_config.bcs['INFLOW'].Q = [q / ((len(lpa_info.values()) + len(rpa_info.values())) // 2) for q in pa_config.bcs['INFLOW'].Q]

    tree_params={'lpa': [19992500, -31.12413547, 0.0, 61.49783421], 
                 'rpa': [19992500, -57.18955744, 0.0, 39.24673482]}

    # create PA config
    pa_config.create_impedance_trees(lpa_mean_dia, rpa_mean_dia, 0.01, tree_params, n_procs=24)

    pa_config.simulate()

    print(f'pa config SIMULATED, rpa split: {pa_config.rpa_split}, p_mpa = {pa_config.P_mpa}\n')

    pa_config.to_json(f'pa_config_rcr_comparison.json')

    pa_config.plot_mpa()

    pa_config.optimize_rcrs_and_compare()


def optimize_rcrs():
    '''
    optimize rcr bcs to clinical targets'''

    os.chdir('cases/zerod/impedance/pa_tuning/nonlin')
    zerod_config_path = 'simplified_nonlin_zerod.json'

    msh_surf_path = '../../../../../../threed_models/SU0243/preop/mesh-complete/mesh-surfaces'

    config_handler = ConfigHandler.from_json(zerod_config_path, is_pulmonary=True)

    clinical_targets = ClinicalTargets(wedge_p=5.0, mpa_p=[72.0, 14.0, 38.0], rpa_split=0.8)

    # get the rpa/lpa info
    rpa_info, lpa_info, inflow_info = vtp_info(msh_surf_path, convert_to_cm=True, pulmonary=True)
    
    # rpa_mean_dia = 0.32
    rpa_mean_dia = np.mean([(area / np.pi)**(1/2) * 2 for area in rpa_info.values()])
    print(f'RPA mean diameter: {rpa_mean_dia}')
    # lpa_mean_dia = 0.32
    lpa_mean_dia = np.mean([(area / np.pi)**(1/2) * 2 for area in lpa_info.values()])
    print(f'LPA mean diameter: {lpa_mean_dia}')
    
    pa_config = PAConfig.from_pa_config(config_handler, clinical_targets)

    # rescale inflow by number of outlets ## TODO: figure out scaling for this
    pa_config.bcs['INFLOW'].Q = [q / ((len(lpa_info.values()) + len(rpa_info.values())) // 2) for q in pa_config.bcs['INFLOW'].Q]

    def loss_function(params, pa_config, clinical_targets):
        '''
        params: [rp_lpa, c_lpa, rd_lpa, rp_rpa, c_rpa, rd_rpa]
        '''

        pa_config.bcs["LPA_BC"] = BoundaryCondition.from_config({
            "bc_name": "LPA_BC",
            "bc_type": "RCR",
            "bc_values": {
                "Rp": 0.1 * params[0],
                "C": params[1],
                "Rd": 0.9 * params[0],
                "Pd": clinical_targets.wedge_p
            }
        })

        pa_config.bcs["RPA_BC"] = BoundaryCondition.from_config({
            "bc_name": "RPA_BC",
            "bc_type": "RCR",
            "bc_values": {
                "Rp": 0.1 * params[2],
                "C": params[3],
                "Rd": 0.9 * params[2],
                "Pd": clinical_targets.wedge_p
            }
        })

        pa_config.simulate()

        rpa_split = pa_config.rpa_split

        P_mpa = pa_config.P_mpa

        pressure_loss = np.sum(np.dot(np.abs(np.array(pa_config.P_mpa) - np.array(clinical_targets.mpa_p)), np.array([1, 10, 1]))) ** 2

        flowsplit_loss = ((pa_config.rpa_split - clinical_targets.rpa_split) * 100) ** 2

        loss = pressure_loss + flowsplit_loss
        print(f'Loss: {loss}')

        return loss
    
    bounds = Bounds(lb=[0.0, 0.0, 0.0, 0.0], ub=[np.inf, np.inf, np.inf, np.inf])

    res = minimize(loss_function, [100000.0, 0.001, 100000.0, 0.001], args=(pa_config, clinical_targets), method='Nelder-Mead', bounds=bounds)

    pa_config.plot_mpa('mpa_plot_rcr.png')
    print(f'Optimized RCRs: {res.x}')
    print(f'rpa flow split: {pa_config.rpa_split}, mpa pressure: {pa_config.P_mpa}')


def impedance_tree_sweep(param1, param2, computed=False):
    '''
    plot the poiseuille resistance (Z(w=0)) for the impedance tree at different values of alpha, beta, xi, k2, k1, k3, lrr
    '''

    os.chdir('cases/zerod/impedance/tree_params')

    print(f'creating Z(w=0) surface for {param1}, {param2}')

    params = {
        'd_root': [0.05, 0.1, 0.15, 0.2, 0.25],
        'd_min': [0.01, 0.02, 0.03, 0.04, 0.05],
        'alpha': [0.5, 0.6, 0.7, 0.8, 0.9],
        'beta': [0.4, 0.5, 0.6, 0.7, 0.8],
        'xi': [1.0, 1.5, 2.0, 2.5, 3.0],
        'k1': [1e6, 5e6, 1e7, 5e7, 1e8],
        'k2': [-65, -55, -45, -35, -25],
        'k3': [1e4, 1e5, 1e6, 1e7, 1e8],
        'lrr': [10.0, 20.0, 30.0, 40.0, 50.0]
    }

    def compute_z0(d_root=0.25, d_min=0.01, alpha=0.9, beta=0.6, xi=2.7, k1=2e7, k2=-35, k3=0.0, lrr=50.0, **kwargs):
        '''
        compute the impedance at w=0
        '''
        # create a dictionary of the parameters

        print('using kwargs: ', kwargs)



        # create a structured tree
        tree = StructuredTree(name='test_impedance', time=np.linspace(0, 1.0, 512), Pd=5.0, simparams=None)
        tree.build_tree(d_root, d_min, alpha=alpha, beta=beta, xi=xi, lrr=lrr)

        # compute Z(w=0)
        Z = tree.root.z0_olufsen(0.0, k1=k1, k2=k2, k3=k3)

        return Z

    # make meshgrid of alpha and beta and evaluate z0 on the mesh
    param1_vals, param2_vals = np.meshgrid(params[param1], params[param2])
    if computed:
        with open(f'{param1}_{param2}_z0.pkl', 'rb') as f:
            z0 = pickle.load(f)
    else:
        z0 = np.zeros((len(param1_vals), len(param2_vals)))
        count = 0
        total_count = len(param1_vals) * len(param2_vals)
        for i in range(len(params[param1])):
            for j in range(len(params[param2])):
                count += 1
                print(f'Computing z0: {count}/{total_count}')
                z0[i, j] = compute_z0(**{param1: param1_vals[i, j], param2: param2_vals[i, j]})

        # save the results to a pickle file
        with open(f'{param1}_{param2}_z0.pkl', 'wb') as f:
            pickle.dump(z0, f)
    
    # 3D plot of the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(param1_vals, param2_vals, z0, cmap='viridis')
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel('Z(w=0)')
    ax.set_title(f'Z(w=0) vs {param1} and {param2}')
    plt.savefig(f'{param1}_{param2}_z0.png')
    plt.show()

def test_pa_config(build_tree=True):
    '''
    '''

    os.chdir('cases/zerod/impedance/pa_tuning/nonlin/t2000')

    clinical_targets = ClinicalTargets(wedge_p=5.0, mpa_p=[72.0, 14.0, 38.0], rpa_split=0.8)


    config_handler = ConfigHandler.from_json('simplified_nonlin_zerod.json', is_pulmonary=True)
    pa_config = PAConfig.from_pa_config(config_handler, clinical_targets)
    pa_config.create_impedance_trees(0.32, 0.32, [0.01, 0.01], {'lpa': [19992500, -31.12413547, 0.0, 61.49783421, 0.9, 0.6],
                                                        'rpa': [19992500, -57.18955744, 0.0, 39.24673482, 0.9, 0.6]}, n_procs=24)
        

    # print stenosis coeff for all vessels
    print([vessel.stenosis_coefficient for vessel in pa_config.vessel_map.values()])

    pa_config.simulate()
    print(pa_config.rpa_split)
    pa_config.plot_mpa(path=None)
    pa_config.plot_outlets(path=None)

    pa_config.to_json('pa_config_test.json')


def test_stenosis_coefficient():
    config = ConfigHandler.from_json('cases/zerod/simple_config/simple_config_1out.json', is_pulmonary=False)

    # print(config.config)
    result = pysvzerod.simulate(config.config)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    t = result[result.name=='branch0_seg0']['time']
    p = result[result.name=='branch0_seg0']['pressure_in'] / 1333.2
    q = result[result.name=='branch0_seg0']['flow_in']


    print(f'p end: {p.iloc[-1]}, q end: {q.iloc[-1]}')
    # # flow plot
    # axs[0].plot(t, q, label='0D')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].set_ylabel('Flow (mL/s)')

    # # pressure
    # axs[1].plot(t, p, label='0D')
    # axs[1].set_xlabel('Time (s)')
    # axs[1].set_ylabel('Pressure (mmHg)')

    # plt.show()

    

if __name__ == '__main__':
    # test_impedance_tree(build_tree=True)
    
    # tune_tree()

    # run_optimized_tree()

    test_tune_pa_trees()

    # test_construct_impedance_trees()

    # compare_3d_0d_outlet()

    # test_rcr_comparison()

    # optimize_rcrs()

    # impedance_tree_sweep('k2', 'lrr', False)

    # test_pa_config(False)

    # test_stenosis_coefficient()

