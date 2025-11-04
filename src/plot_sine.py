import csv
import cmath
import numpy as np
import matplotlib.pyplot as plt
import random

from math import sqrt
from tags.state_machine import StateSerializer
from main import load_json, load_txt
from tags.tag import TagMode, Tag
from physics import PhysicsEngine 
from state import AppState


OUTPUT_CSV = "feedback_phase_sine.csv"
CONFIG_PATH = "demo/3_tag_test/config_files/config.json"

# --- Generate 36 discrete chip impedances (10° increments) ---
Z_ant = 50 + 0j            # assume 50Ω antenna impedance
avoid_infinite = 1e6 + 0j   # used for 0° open-circuit
N = 36                      # 10° increments → 36 steps

def gamma_from_deg(deg):
    """Return reflection coefficient for a given phase angle in degrees."""
    return cmath.exp(1j * np.deg2rad(deg))

def zchip_from_gamma(gamma, Z_ant=Z_ant):
    """Compute chip impedance that yields reflection coefficient gamma."""
    # Handle 0° case (Γ=1) → open circuit
    if abs(abs(gamma) - 1.0) < 1e-12 and abs(np.angle(gamma)) < 1e-9:
        return avoid_infinite
    return Z_ant * (1 + gamma) / (1 - gamma)

def voltage_at_tag_NoFL(self, tags, receiving_tag):
    """
    Get the total voltage delivered to a given tag WITHOUT feedback loops.
    This is a simplified version that doesn't account for backscatter feedback.

    Parameters:
        physics_engine (PhysicsEngine): The physics engine instance.
        tags (dict[str, Tag]): A dictionary of all the tags in the simulation.
        receiving_tag (Tag): The tag to get the voltage for.
    Returns:
        float: The voltage at the receiving tag's envelope detector input.
    """
    rx_impedance = receiving_tag.get_impedance()
    sigs_to_rx = []

    # Loop over ALL exciters (now a dict)
    for ex in self.exciters.values():
        # Direct exciter → receiver
        sigs_to_rx.append(self.get_sig_tx_rx(ex, receiving_tag))

        # Backscatter from other tags (first-order only, no feedback)
        for tag in tags.values():
            if tag is receiving_tag:
                continue

            reflection_coeff = self.effective_reflection_coefficient(tag)
            if abs(reflection_coeff) < 1e-6:
                continue

            sig_ex_tx = self.get_sig_tx_rx(ex, tag)
            sig_tx_rx = self.get_sig_tx_rx(tag, receiving_tag)
            sigs_to_rx.append(sig_ex_tx * reflection_coeff * sig_tx_rx)

    # Combine signals
    pwr_received = abs(sum(sigs_to_rx))
    v_pk = sqrt(abs(rx_impedance * pwr_received) / 500.0)
    v_rms = v_pk / sqrt(2.0)

    # Add optional AWGN noise
    if self.noise_std_volts and self.noise_std_volts > 0.0:
        v_rms = max(0.0, random.gauss(v_rms, self.noise_std_volts))

    return v_rms

def phase_sweep_NoFL():
    app_state = AppState()
    serializer = StateSerializer()

    # Load 3-tag test case config
    exciter, tags, _, _ = load_json(CONFIG_PATH, serializer, app_state=app_state)

    phases = np.linspace(0, 360, N, endpoint=False)  # 0°,10°,20°,...,350°
    chip_impedances = [zchip_from_gamma(gamma_from_deg(p)) for p in phases]

    # Initialize physics engine
    physics_engine = PhysicsEngine(exciter, tags)

    for tag in tags.values():
        tag.chip_impedances = chip_impedances

    tx1 = tags["TX1"]
    tx2 = tags["TX2"]
    rx = tags["RX1"]

    results = []

    phase_shifts = {i: (i * 10) % 360 for i in range(N)}

    num_modes_tx1 = len(tx1.chip_impedances)
    num_modes_tx2 = len(tx2.chip_impedances)

    for i in range(num_modes_tx1):
        tx1.set_mode(TagMode(i))
        for j in range(num_modes_tx2):
            tx2.set_mode(TagMode(j))

            v_rx = voltage_at_tag_NoFL(physics_engine, tags, rx)

            results.append({
                "tx1_mode": i,
                "tx2_mode": j,
                "tx1_phase": phase_shifts.get(i, 0),
                "tx2_phase": phase_shifts.get(j, 0),
                "v_rx_volts": v_rx
            })

            print(f"TX1={i}, TX2={j} → Vrx={v_rx:.6f} V")

    return results

def run_feedback_phase_sweep():
    app_state = AppState()
    serializer = StateSerializer()

    # Load 3-tag test case config
    exciter, tags, _, _ = load_json(CONFIG_PATH, serializer, app_state=app_state)

    phases = np.linspace(0, 360, N, endpoint=False)  # 0°,10°,20°,...,350°
    chip_impedances = [zchip_from_gamma(gamma_from_deg(p)) for p in phases]

    # Initialize physics engine
    physics_engine = PhysicsEngine(exciter, tags)

    for tag in tags.values():
        tag.chip_impedances = chip_impedances

    tx1 = tags["TX1"]
    tx2 = tags["TX2"]
    rx = tags["RX1"]

    results = []

    phase_shifts = {i: (i * 10) % 360 for i in range(N)}

    num_modes_tx1 = len(tx1.chip_impedances)
    num_modes_tx2 = len(tx2.chip_impedances)

    for i in range(num_modes_tx1):
        tx1.set_mode(TagMode(i))
        for j in range(num_modes_tx2):
            tx2.set_mode(TagMode(j))

            # Debug prints
            print(f"\nTX1 (mode {i}) reflection: {physics_engine.effective_reflection_coefficient(tx1)}")
            print(f"TX2 (mode {j}) reflection: {physics_engine.effective_reflection_coefficient(tx2)}")
            print(f"TX1 powered: {physics_engine.is_tag_powered(tx1)}")
            print(f"TX2 powered: {physics_engine.is_tag_powered(tx2)}")

            v_rx = physics_engine.voltage_at_tag(tags, rx)

            # Calculate total phase at RX from both transmitters
            sig1 = physics_engine.get_sig_tx_rx(tx1, rx)
            sig2 = physics_engine.get_sig_tx_rx(tx2, rx)
            total_sig = sig1 + sig2
            total_phase = np.angle(total_sig, deg=True)

            results.append({
                "tx1_mode": i,
                "tx2_mode": j,
                "tx1_phase": phase_shifts.get(i, 0),
                "tx2_phase": phase_shifts.get(j, 0),
                "v_rx_volts": v_rx,
                "total_phase": total_phase
            })

            print(f"TX1={i}, TX2={j} → Vrx={v_rx:.6f} V")

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["tx1_mode","tx2_mode","tx1_phase","tx2_phase","v_rx_volts","total_phase"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")
    return results

def plot_results(results_FL, results_NoFL):
    """
    Plot RX voltage for each TX mode combination, comparing feedback vs no feedback.
    """
    labels = [f"{r['tx1_phase']}°,{r['tx2_phase']}°" for r in results_FL]
    values_FL = [r["v_rx_volts"] for r in results_FL]
    values_NoFL = [r["v_rx_volts"] for r in results_NoFL]

    plt.figure(figsize=(14, 6))
    x_indices = range(len(labels))
    
    plt.plot(x_indices, values_FL, marker='o', markersize=3, linestyle='-', 
             label="With Feedback Loop", linewidth=1.5, alpha=0.8)
    plt.plot(x_indices, values_NoFL, marker='s', markersize=3, linestyle='--', 
             label="No Feedback Loop", linewidth=1.5, alpha=0.8)
    
    plt.xlabel("TX Mode Combination Index (TX1°, TX2°)")
    plt.ylabel("RX Voltage (Volts)")
    plt.title("RX Voltage for TX Mode Combinations: Feedback vs No Feedback")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show every 36th label (once per full TX1 cycle)
    step = 36
    plt.xticks([i for i in x_indices if i % step == 0], 
               [labels[i] for i in x_indices if i % step == 0], 
               rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running feedback phase sweep...")
    results_FL = run_feedback_phase_sweep()
    
    print("\nRunning no-feedback phase sweep...")
    results_NoFL = phase_sweep_NoFL()
    
    print("\nPlotting results...")
    plot_results(results_FL, results_NoFL)