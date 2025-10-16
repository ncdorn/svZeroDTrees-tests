import os
import json
import pysvzerod
# import svzerodtrees
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.integrate import trapezoid

def test_right_heart_simulation():
    '''
    Test the right heart simulation setup and execution.
    '''
    # Define paths
    casename = 'reg_chamber_pa'
    config_path = 'cases/zerod/rh_chamber/' + casename + '.json'
    simulation_dir = 'cases/zerod/rh_chamber/results/'

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Ensure simulation directory exists
    os.makedirs(simulation_dir, exist_ok=True)

    # Initialize simulation
    result = pysvzerod.simulate(config)

    results_path = os.path.join(simulation_dir, casename + '_results.csv')

    result.to_csv(results_path)

    # --- Config ---
    CSV_PATH = Path(results_path)  # update if needed
    df = pd.read_csv(CSV_PATH)

    def get_trace(signal):
        sub = df[df["name"] == signal].sort_values("time")
        if sub.empty:
            raise KeyError(f"Signal {signal} not found. Available: {df['name'].unique()}")
        return sub["time"].values, sub["y"].values

    # --- Signals ---
    signals = {
        "atrial_pressure": "pressure:INLET:valve0",
        "ventricle_pressure": "pressure:ventricle:valve1",
        "outlet_pressure": "pressure:valve1:vessel",
        # "outlet_pressure": "pressure:valve1:vessel",
        "atrial_flow": "flow:INLET:valve0",
        "ventricle_flow": "flow:ventricle:valve1",
        "outlet_flow": "flow:valve1:vessel",
        "ventricle_volume": "Vc:ventricle",
    }

    # Extract traces
    t_ap, ap = get_trace(signals["atrial_pressure"])
    t_vp, vp = get_trace(signals["ventricle_pressure"])
    t_op, op = get_trace(signals["outlet_pressure"])
    t_af, af = get_trace(signals["atrial_flow"])
    t_vf, vf = get_trace(signals["ventricle_flow"])
    t_of, of = get_trace(signals["outlet_flow"])
    t_vc, vc = get_trace(signals["ventricle_volume"])

    # --- Cardiac Output calculation ---
    dt = np.mean(np.diff(t_of))

    # Guess cycle length: last time / number of cycles (from config)
    T_cycle = t_vf.max()
    n_pts_per_cycle = int(round(T_cycle / dt))

    vf_cycle = vf[-n_pts_per_cycle:]
    t_cycle = t_vf[-n_pts_per_cycle:]

    SV = trapezoid(vf_cycle, t_cycle)     # cm³ = mL stroke volume
    HR = 60.0 / T_cycle                  # beats per min
    CO_ml_min = SV * HR                  # mL/min
    CO_L_min = CO_ml_min / 1000.0        # L/min

    # --- Plot 2x4 grid ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)

    # Top row: pressures + volume
    axes[0,0].plot(t_ap, ap / 1333.2); axes[0,0].set_title("RA Pressure"); axes[0,0].set_xlabel("Time [s]"); axes[0,0].set_ylabel("mmHg")
    axes[0,1].plot(t_vp, vp / 1333.2); axes[0,1].set_title("RV Pressure"); axes[0,1].set_xlabel("Time [s]"); axes[0,1].set_ylabel("mmHg")
    axes[0,2].plot(t_op, op / 1333.2); axes[0,2].set_title("MPA Pressure"); axes[0,2].set_xlabel("Time [s]"); axes[0,2].set_ylabel("mmHg")
    axes[0,3].plot(t_vc, vc); axes[0,3].set_title("Ventricle Volume"); axes[0,3].set_xlabel("Time [s]"); axes[0,3].set_ylabel("mL")

    # Bottom row: flows + PV loop
    axes[1,0].plot(t_af, af); axes[1,0].set_title("RA Flow"); axes[1,0].set_xlabel("Time [s]"); axes[1,0].set_ylabel("mL/s")
    axes[1,1].plot(t_vf, vf); axes[1,1].set_title("RV Flow"); axes[1,1].set_xlabel("Time [s]"); axes[1,1].set_ylabel("mL/s")
    axes[1,2].plot(t_of, of); axes[1,2].set_title("MPA Flow"); axes[1,2].set_xlabel("Time [s]"); axes[1,2].set_ylabel("mL/s")
    axes[1,3].plot(vc, vp / 1333.2); axes[1,3].set_title("RV PV Loop"); axes[1,3].set_xlabel("Volume [mL]"); axes[1,3].set_ylabel("Pressure [mmHg]")

    plt.suptitle(f"Chamber + Valve Hemodynamics — Cardiac Output = {CO_L_min:.2f} L/min", fontsize=16)
    # Save and/or show
    out_png = simulation_dir + casename + '_results.png'
    plt.savefig(out_png, dpi=200)
    print(f"Saved figure to: {out_png}")

    # If running interactively, uncomment the next line:
    # plt.show()



if __name__ == "__main__":
    test_right_heart_simulation()