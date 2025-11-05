import csv
import cmath
import numpy as np
import matplotlib.pyplot as plt
import random

import os
import sys
# Add project_root/src to sys.path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from math import sqrt
from tags.state_machine import StateSerializer
from main import load_json
from tags.tag import TagMode, Tag
from physics import PhysicsEngine 
from state import AppState

OUTPUT_CSV = "feedback_phase_sine.csv"
CONFIG_PATH = "demo/3_tag_test/config_files/config.json"

# --- Generate 36 discrete chip impedances (10° increments) ---
Z_ant = 50 + 0j
avoid_infinite = 1e6 + 0j
N = 36  # 0°,10°,...,350°

def gamma_from_deg(deg):
    return cmath.exp(1j * np.deg2rad(deg))

def zchip_from_gamma(gamma, Z_ant=Z_ant):
    if abs(abs(gamma) - 1.0) < 1e-12 and abs(np.angle(gamma)) < 1e-9:
        return avoid_infinite
    return Z_ant * (1 + gamma) / (1 - gamma)

def voltage_at_tag_NoFL(self, tags, receiving_tag):
    """Voltage at RX without feedback loops."""
    rx_impedance = receiving_tag.get_impedance()
    sigs_to_rx = []
    for ex in self.exciters.values():
        sigs_to_rx.append(self.get_sig_tx_rx(ex, receiving_tag))
        for tag in tags.values():
            if tag is receiving_tag:
                continue
            reflection_coeff = self.effective_reflection_coefficient(tag)
            if abs(reflection_coeff) < 1e-6:
                continue
            sig_ex_tx = self.get_sig_tx_rx(ex, tag)
            sig_tx_rx = self.get_sig_tx_rx(tag, receiving_tag)
            sigs_to_rx.append(sig_ex_tx * reflection_coeff * sig_tx_rx)

    pwr_received = abs(sum(sigs_to_rx))
    v_pk = sqrt(abs(rx_impedance * pwr_received) / 500.0)
    v_rms = v_pk / sqrt(2.0)
    if self.noise_std_volts and self.noise_std_volts > 0.0:
        v_rms = max(0.0, random.gauss(v_rms, self.noise_std_volts))
    return v_rms

def phase_sweep_NoFL():
    app_state = AppState()
    serializer = StateSerializer()
    exciters, tags, _, _ = load_json(CONFIG_PATH, serializer, app_state=app_state)
    phases = np.linspace(0, 360, N, endpoint=False)
    chip_impedances = [zchip_from_gamma(gamma_from_deg(p)) for p in phases]
    physics_engine = PhysicsEngine(exciters)
    for tag in tags.values():
        tag.chip_impedances = chip_impedances
    tx1, tx2, rx = tags["TX1"], tags["TX2"], tags["RX1"]

    results = []
    for i, phi1 in enumerate(phases):
        tx1.set_mode(TagMode(i))
        for j, phi2 in enumerate(phases):
            tx2.set_mode(TagMode(j))
            v_rx = voltage_at_tag_NoFL(physics_engine, tags, rx)
            results.append({"tx1_phase": phi1, "tx2_phase": phi2, "v_rx_volts": v_rx})
    return results

def run_feedback_phase_sweep():
    app_state = AppState()
    serializer = StateSerializer()
    exciters, tags, _, _ = load_json(CONFIG_PATH, serializer, app_state=app_state)
    phases = np.linspace(0, 360, N, endpoint=False)
    chip_impedances = [zchip_from_gamma(gamma_from_deg(p)) for p in phases]
    physics_engine = PhysicsEngine(exciters)
    for tag in tags.values():
        tag.chip_impedances = chip_impedances
    tx1, tx2, rx = tags["TX1"], tags["TX2"], tags["RX1"]

    results = []
    for i, phi1 in enumerate(phases):
        tx1.set_mode(TagMode(i))
        for j, phi2 in enumerate(phases):
            tx2.set_mode(TagMode(j))
            v_rx = physics_engine.voltage_at_tag(tags, rx)
            results.append({"tx1_phase": phi1, "tx2_phase": phi2, "v_rx_volts": v_rx})
    return results

# ---------- MODULATION DEPTH ----------
def compute_modulation_depth(results):
    """Compute |V2 - V1| / max(V1, V2) * 100% for consecutive TX1 phases."""
    grouped = {}
    for r in results:
        phi1 = r["tx1_phase"]
        grouped.setdefault(phi1, []).append(r["v_rx_volts"])
    phases = sorted(grouped.keys())
    mod_depths = []
    for i in range(len(phases) - 1):
        v1 = np.mean(grouped[phases[i]])
        v2 = np.mean(grouped[phases[i+1]])
        M = abs(v2 - v1) / max(v1, v2) * 100
        mod_depths.append({"phase_pair": (phases[i], phases[i+1]), "M_percent": M})
    avg_M = np.mean([m["M_percent"] for m in mod_depths])
    return mod_depths, avg_M

# ---------- PLOTTING ----------
def plot_results(results_FL, results_NoFL, mod_FL, mod_NoFL):
    labels = [f"{r['tx1_phase']}°, {r['tx2_phase']}°" for r in results_FL]
    v_FL = [r["v_rx_volts"] for r in results_FL]
    v_NoFL = [r["v_rx_volts"] for r in results_NoFL]

    plt.figure(figsize=(14, 6))
    x = np.arange(len(labels))
    plt.plot(x, v_FL, label="Feedback Loop", lw=1.5)
    plt.plot(x, v_NoFL, label="No Feedback", lw=1.5, ls="--")
    plt.xlabel("TX1°,TX2° Index")
    plt.ylabel("RX Voltage (V)")
    plt.title("RX Voltage vs Phase Combination")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Modulation depth plot
    plt.figure(figsize=(10, 4))
    plt.plot([f"{int(a)}–{int(b)}°" for (a,b) in [m["phase_pair"] for m in mod_FL]],
             [m["M_percent"] for m in mod_FL],
             label="Feedback", marker='o')
    plt.plot([f"{int(a)}–{int(b)}°" for (a,b) in [m["phase_pair"] for m in mod_NoFL]],
             [m["M_percent"] for m in mod_NoFL],
             label="No Feedback", marker='s')
    plt.xticks(rotation=45)
    plt.ylabel("Modulation Depth (%)")
    plt.title("Modulation Depth per Phase Step")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running feedback phase sweep...")
    results_FL = run_feedback_phase_sweep()
    print("Running no-feedback phase sweep...")
    results_NoFL = phase_sweep_NoFL()

    mod_FL, avg_FL = compute_modulation_depth(results_FL)
    mod_NoFL, avg_NoFL = compute_modulation_depth(results_NoFL)
    increase_pct = (avg_FL - avg_NoFL) / avg_NoFL * 100

    print(f"\nAverage Modulation Depth (Feedback): {avg_FL:.2f}%")
    print(f"Average Modulation Depth (No Feedback): {avg_NoFL:.2f}%")
    print(f"Feedback increases modulation by {increase_pct:.2f}%")

    plot_results(results_FL, results_NoFL, mod_FL, mod_NoFL)
