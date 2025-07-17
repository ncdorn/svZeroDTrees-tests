import numpy as np
import pandas as pd
# print(sys.path)
from pathlib import Path
import scipy
from scipy.integrate import solve_ivp
from svzerodtrees.microvasculature.utils import *
from svzerodtrees.adaptation.experiment import *
from svzerodtrees.adaptation import *
from svzerodtrees import *


# this function is not great and will lively be replaced at some point.
def post_process_solution(postop_pa, flow_log, sol):
    # helper for post processing
    def radius_trace(v):
        """Return the time-series of radius (meters) for vessel v."""
        return sol.y[2 * v.idx]                      # radii are at even indices
    
    all_vessels = postop_pa.lpa_tree.enumerate_vessels() + postop_pa.rpa_tree.enumerate_vessels(start_idx=postop_pa.lpa_tree.count_vessels())

    # choose vessels to probe - [lpa root, lpa last vessel, rpa root, rpa last vessel]
    probe_idxs = [0, postop_pa.lpa_tree.count_vessels()-1, postop_pa.lpa_tree.count_vessels(), len(all_vessels) - 1]               # edit as needed
    probe_vessels = [v for v in all_vessels if v.idx in probe_idxs]

    # save radius
    df = pd.DataFrame({
        "t_s"   : sol.t,
    })

    for v in probe_vessels:
        df[f"r_{v.idx}_mm"] = radius_trace(v)

    df.to_csv("radius_trace.csv", index=False)
    print("✓ saved radius_trace.csv")

    # save flow split
    t_vals, phi_vals = map(np.asarray, zip(*flow_log))
    phi_df = pd.DataFrame({"t_s": t_vals, "phi": phi_vals})
    phi_df.to_csv("flow_split_trace.csv", index=False)
    print("✓  saved flow_split_trace.csv")

    # save numpy compressed
    np.savez_compressed(
        "adaptation_raw.npz",
        t = sol.t.astype(np.float32),
        y = sol.y.astype(np.float32)
    )
    print("✓ saved adaptation_raw.npz")


# with radau, this converges at 66.6% for d_min 0.05
# if we bump up K_tau_r and K_sig_r to 5e-7, we get to 60.6% for d_min 0.05, pre adaptation 68.9%
# if we use K_r / k_h = 10, we get 57.4%
# if we use K_r / k_h = 5, we get 64.9

def generate_postop_reduced_config(preop_simdir_path, postop_simdir_path):
    preop_simdir = simulation.SimulationDirectory.from_directory(preop_simdir_path, convert_to_cm=True)
    postop_simdir = simulation.SimulationDirectory.from_directory(postop_simdir_path, convert_to_cm=True)

    # postop_simdir.optimize_nonlinear_resistance("cases/threed/SU0243/pa_config_test_tuning.json")

    S_lpa_preop, S_rpa_preop = preop_simdir.compute_pressure_drop(steady=False)
    

    S_lpa_postop, S_rpa_postop = postop_simdir.compute_pressure_drop(steady=False)

    print(f"Preop LPA resistance: {S_lpa_preop:.2f}")
    print(f"Preop RPA resistance: {S_rpa_preop:.2f}")

    print(f"Postop LPA resistance: {S_lpa_postop:.2f}")
    print(f"Postop RPA resistance: {S_rpa_postop:.2f}")

    print("postop lpa resistance reduction: ", S_lpa_postop / S_lpa_preop)


def test_adapt_cwss_ims_integration_single():

    results_df = run_single_gain_cwss_ims_adaptation(
        preop_config_path = "cases/zerod/tree-adaptation/simple_pa/preop_pa_config.json",
        postop_config_path = "cases/zerod/tree-adaptation/simple_pa/postop_pa_config.json",
        optimized_tree_params_csv = "cases/threed/SU0243/optimized_params.csv",
        clinical_targets_csv= "cases/threed/SU0243/clinical_targets.csv",
        gain = 1e-7
    )

    filename = "single_gains_lpa_perturbed.csv"
    results_df.to_csv(filename, index=False)
    print(f"✓ saved {filename}")

def test_adapt_cwss_ims_integration_all():

    relatives = [0, 1, 10]  # includes zeros
    base_gain = 1e-7

    results_df = run_parallel_gain_combinations(
        relatives=relatives,
        base_gain=base_gain,
        preop_config_path="cases/zerod/tree-adaptation/simple_pa/preop_pa_config.json",
        postop_config_path="cases/zerod/tree-adaptation/simple_pa/postop_pa_config.json",
        optimized_tree_params_csv="cases/threed/SU0243/optimized_params.csv",
        clinical_targets_csv="cases/threed/SU0243/clinical_targets.csv",
        max_workers=8,
        combinations_csv_path= "cases/zerod/tree-adaptation/gain_combinations.csv",
    )

    print("Sweep complete.")

    filename = "all_gains_lpa_stent.csv"
    results_df.to_csv(filename, index=False)
    print(f"✓ saved {filename}")

def test_finer_gain_combos():

    gain_sets = [ # from chatGPT
    [1e-7, 1e-6, 1e-7, 1e-7],        # A
    [1e-7, 1e-6, 1e-6, 1e-7],        # B
    [1e-7, 5e-6, 1e-7, 1e-7],        # A
    [1e-7, 1e-5, 1e-7, 1e-7],        # A
    ]

    results_df = run_parallel_gains(
        scaled_K_arrs=gain_sets,
        preop_config_path="cases/zerod/tree-adaptation/simple_pa/preop_pa_config.json",
        postop_config_path="cases/zerod/tree-adaptation/simple_pa/postop_pa_config.json",
        optimized_tree_params_csv="cases/threed/SU0243/optimized_params.csv",
        clinical_targets_csv="cases/threed/SU0243/clinical_targets.csv",
        max_workers=len(gain_sets),
        combinations_csv_path= "cases/zerod/tree-adaptation/gain_combinations.csv",
    )

    existing_csv = Path('gains_lpa_stent_dmin05.csv')

    # ---------------------------------------------------------------------
    # 1. Load current results (if the file is already present)
    # ---------------------------------------------------------------------
    if existing_csv.is_file():
        cumulative = pd.read_csv(existing_csv)
    else:
        cumulative = pd.DataFrame(columns=results_df.columns)  # empty shell

    # ---------------------------------------------------------------------
    # 2. Concatenate & optionally de-duplicate
    # ---------------------------------------------------------------------
    updated = pd.concat([cumulative, results_df], ignore_index=True)

    # If you want to drop exact duplicates of *all* columns, uncomment:
    # updated = updated.drop_duplicates(keep='last')

    # ---------------------------------------------------------------------
    # 3. Save back to disk
    # ---------------------------------------------------------------------
    updated.to_csv(existing_csv, index=False)
    print(f"Appended {len(results_df)} rows → {existing_csv} "
          f"(total rows now: {len(updated)})")

def test_adaptation_threed():
    '''
    test the microvascular adaptation with a 3D result
    '''

    preop_simdir = SimulationDirectory.from_directory("cases/threed/SU0243/preop", convert_to_cm=True)
    postop_simdir = SimulationDirectory.from_directory("cases/threed/SU0243/postop", convert_to_cm=True)
    adapted_simdir = SimulationDirectory.from_directory("cases/threed/SU0243/adapted-cwss-ims", convert_to_cm=True)
    reduced_order_pa = "cases/zerod/tree-adaptation/simple_pa/preop_pa_config.json"
    tree_params = "cases/threed/SU0243/optimized_params.csv"
    clinical_targets = ClinicalTargets.from_csv("cases/threed/SU0243/clinical_targets.csv")

    microvascular_adaptor = MicrovascularAdaptor(
        preop_simdir=preop_simdir,
        postop_simdir=postop_simdir,
        adapted_simdir=adapted_simdir,
        reduced_order_pa=reduced_order_pa,
        tree_params_csv=tree_params,
        clinical_targets=clinical_targets,
        convert_to_cm=True
    )

    K_arr = [1e-7, 1e-7, 1e-7, 1e-7]  # example gain array

    microvascular_adaptor.adapt_cwss_ims(K_arr)


if __name__ == "__main__":
    # run the test
    # generate_postop_reduced_config("cases/threed/SU0243/preop", "cases/threed/SU0243/postop")
    # test_adapt_cwss_ims_integration_all()
    # print("Adaptation investigation completed.")

    test_finer_gain_combos()