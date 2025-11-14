import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
import cmath

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tags.state_machine import StateSerializer
from tags.tag import TagMode
from main import load_json
from physics import PhysicsEngine
from state import AppState


# ---------------- CONFIG ----------------
CONFIG_PATH = "demo/7_tag_test/config_files/config2.json"
NUM_COEFFS = 6
USE_JITTER = True
USE_HISTOGRAM = True
NUM_TRAINING_SAMPLES = 50  # Number of measurements for curve fitting
NUM_TOP_PREDICTIONS = 10   # How many top predictions to verify
# ----------------------------------------


def mW_to_mV(v):
    """Helper: convert volts to millivolts."""
    return v * 1000.0


def compute_modulation_depth(physics_engine, tags, tx, rx):
    """Compute modulation depth with current tag configuration."""
    tx.set_mode(TagMode(1))
    v_high = physics_engine.voltage_at_tag(tags, rx)
    tx.set_mode(TagMode(2))
    v_low = physics_engine.voltage_at_tag(tags, rx)
    return mW_to_mV(abs(v_high - v_low))


def generate_helper_subsets(helper_tags, max_helpers=5):
    """Generate all combinations of helper tags up to given count."""
    subsets = {}
    for k in range(max_helpers + 1):
        subsets[k] = list(itertools.combinations(helper_tags, k))
    return subsets


def get_gamma_set():
    """Get the 7 reflection coefficients (index 0 = non-reflecting, 1-6 = phases)."""
    gammas = [0 + 0j]  # Index 0: non-reflecting
    for theta in [0, 60, 120, 180, 240, 300]:
        gammas.append(cmath.exp(1j * np.deg2rad(theta)))
    return gammas


def model_modulation_depth(gamma_config, *params):
    """
    Model for modulation depth without feedback loops.
    
    Signal model: S_r = |A_e*h_er + A_e*h_et*Γ_t*h_tr + Σ(A_e*h_eh_i*Γ_h_i*h_hr_i)|
    Modulation depth = |S_r1 - S_r0| where TX uses Γ_1 and Γ_0
    
    Args:
        gamma_config: tuple of (tx_gamma_idx_for_0, tx_gamma_idx_for_1, helper_gamma_idx_1, helper_gamma_idx_2, ...)
        params: [A_e, h_er_re, h_er_im, h_et_re, h_et_im, h_tr_re, h_tr_im, 
                 h_eh1_re, h_eh1_im, h_hr1_re, h_hr1_im, ...]
    
    Returns:
        Predicted modulation depth
    """
    gammas = get_gamma_set()
    
    # Extract gamma indices
    tx_gamma_0_idx = int(gamma_config[0])
    tx_gamma_1_idx = int(gamma_config[1])
    helper_gamma_indices = [int(g) for g in gamma_config[2:]]
    num_helpers = len(helper_gamma_indices)
    
    # Extract parameters
    A_e = params[0]
    h_er = complex(params[1], params[2])
    h_et = complex(params[3], params[4])
    h_tr = complex(params[5], params[6])
    
    # Compute S_r0 (TX using gamma_0)
    signal_0 = A_e * h_er
    signal_0 += A_e * h_et * gammas[tx_gamma_0_idx] * h_tr
    
    # Add helper contributions for S_r0
    param_idx = 7
    for i, helper_gamma_idx in enumerate(helper_gamma_indices):
        if helper_gamma_idx == 0:  # Non-reflecting
            param_idx += 4
            continue
        h_eh = complex(params[param_idx], params[param_idx + 1])
        h_hr = complex(params[param_idx + 2], params[param_idx + 3])
        signal_0 += A_e * h_eh * gammas[helper_gamma_idx] * h_hr
        param_idx += 4
    
    # Compute S_r1 (TX using gamma_1)
    signal_1 = A_e * h_er
    signal_1 += A_e * h_et * gammas[tx_gamma_1_idx] * h_tr
    
    # Add same helper contributions for S_r1
    param_idx = 7
    for i, helper_gamma_idx in enumerate(helper_gamma_indices):
        if helper_gamma_idx == 0:
            param_idx += 4
            continue
        h_eh = complex(params[param_idx], params[param_idx + 1])
        h_hr = complex(params[param_idx + 2], params[param_idx + 3])
        signal_1 += A_e * h_eh * gammas[helper_gamma_idx] * h_hr
        param_idx += 4
    
    # Modulation depth
    return abs(abs(signal_1) - abs(signal_0))


def fit_model_parameters(training_data, num_helpers):
    """
    Fit model parameters using training data.
    
    Args:
        training_data: list of (gamma_config, measured_depth) tuples
        num_helpers: number of helpers in the subset
    
    Returns:
        Fitted parameters or None if fitting fails
    """
    if len(training_data) < 10:
        return None
    
    # Prepare data for curve fitting
    X_data = np.array([config for config, _ in training_data])
    y_data = np.array([depth for _, depth in training_data])
    
    # Number of parameters: A_e (1) + h_er (2) + h_et (2) + h_tr (2) + num_helpers * 4
    num_params = 7 + num_helpers * 4
    
    # Initial guess: small positive values
    p0 = np.random.rand(num_params) * 0.1 + 0.01
    
    try:
        # Wrapper function for curve_fit
        def model_wrapper(X, *params):
            return np.array([model_modulation_depth(x, *params) for x in X])
        
        # Use curve_fit with bounds
        bounds_lower = [0.001] + [-10] * (num_params - 1)
        bounds_upper = [10] + [10] * (num_params - 1)
        
        popt, _ = curve_fit(
            model_wrapper, 
            X_data, 
            y_data, 
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        return popt
    except Exception as e:
        print(f"    Curve fitting failed: {e}")
        return None


def heuristic_select_helper_config(physics_engine, tags, tx, rx, helper_subset, 
                                   all_helper_tags, num_training=NUM_TRAINING_SAMPLES):
    """
    Use curve fitting heuristic to estimate best helper configuration.
    
    1. Collect training measurements (~50 random configs)
    2. Fit model parameters using curve_fit
    3. Predict modulation depth for all configs using fitted model
    4. Test top N predictions and return best
    """
    if len(helper_subset) == 0:
        return 0.0, []
    
    num_helpers = len(helper_subset)
    coeff_indices = range(1, NUM_COEFFS + 1)
    all_combos = list(itertools.product(coeff_indices, repeat=num_helpers))
    
    # Phase 1: Collect training data
    training_data = []
    sample_size = min(num_training, len(all_combos))
    sample_indices = np.random.choice(len(all_combos), sample_size, replace=False)
    sample_combos = [all_combos[i] for i in sample_indices]
    
    for combo in sample_combos:
        # Set helper modes
        for helper, coeff_idx in zip(helper_subset, combo):
            helper.set_mode(TagMode(coeff_idx))
        
        # Set non-subset helpers to non-reflecting
        for helper in all_helper_tags:
            if helper not in helper_subset:
                helper.set_mode(TagMode(0))
        
        rx.set_mode(TagMode(0))
        
        # Measure modulation depth
        depth = compute_modulation_depth(physics_engine, tags, tx, rx)
        
        # Store as (gamma_config, depth)
        # gamma_config = (tx_gamma_0, tx_gamma_1, helper1_gamma, helper2_gamma, ...)
        gamma_config = (1, 2) + combo  # TX uses modes 1 and 2
        training_data.append((gamma_config, depth))
    
    # Phase 2: Fit model parameters
    fitted_params = fit_model_parameters(training_data, num_helpers)
    
    if fitted_params is None:
        # Fallback: return best from training data
        best_idx = np.argmax([depth for _, depth in training_data])
        best_depth = training_data[best_idx][1]
        return best_depth, None
    
    # Phase 3: Predict all configurations
    predictions = []
    for combo in all_combos:
        gamma_config = (1, 2) + combo
        predicted_depth = model_modulation_depth(gamma_config, *fitted_params)
        predictions.append((combo, predicted_depth))
    
    # Phase 4: Test top predictions
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:NUM_TOP_PREDICTIONS]
    
    best_depth = 0.0
    best_config = None
    
    for combo, pred_depth in top_predictions:
        # Set helper modes
        for helper, coeff_idx in zip(helper_subset, combo):
            helper.set_mode(TagMode(coeff_idx))
        
        # Set non-subset helpers to non-reflecting
        for helper in all_helper_tags:
            if helper not in helper_subset:
                helper.set_mode(TagMode(0))
        
        rx.set_mode(TagMode(0))
        
        # Measure actual depth
        actual_depth = compute_modulation_depth(physics_engine, tags, tx, rx)
        
        if actual_depth > best_depth:
            best_depth = actual_depth
            best_config = combo
    
    return best_depth, best_config


def run_modulation_depth_sweep():
    app_state = AppState()
    serializer = StateSerializer()

    exciters, tags, _, _ = load_json(CONFIG_PATH, serializer, app_state=app_state)
    physics_engine = PhysicsEngine(exciters)
    tx = tags.get("TX1")
    rx = tags.get("RX1")
    helper_tags = [tag for name, tag in tags.items() if name not in ("TX1", "RX1")]
    
    print(f"Number of helpers: {len(helper_tags)}")
    print(f"Testing with coefficient indices 1-{NUM_COEFFS}")

    all_results = []
    subsets_by_count = generate_helper_subsets(helper_tags, max_helpers=min(5, len(helper_tags)))

    for helper_count, subset_list in subsets_by_count.items():
        print(f"\n=== Evaluating with {helper_count} helpers ===")

        all_depths = []
        subset_best_depths = []
        heuristic_depths = []
        
        for subset_idx, subset in enumerate(subset_list):
            subset_names = [tag.get_name() for tag in subset]
            best_depth_for_subset = 0.0
            
            if helper_count == 0:
                for helper in helper_tags:
                    helper.set_mode(TagMode(0))
                rx.set_mode(TagMode(0))
                
                depth = compute_modulation_depth(physics_engine, tags, tx, rx)
                all_depths.append(depth)
                best_depth_for_subset = depth
                subset_best_depths.append(depth)
                heuristic_depths.append(depth)
            else:
                # Exhaustive search
                coeff_indices = range(1, NUM_COEFFS + 1)
                
                for combo in itertools.product(coeff_indices, repeat=helper_count):
                    for helper, coeff_idx in zip(subset, combo):
                        helper.set_mode(TagMode(coeff_idx))
                    
                    for helper in helper_tags:
                        if helper not in subset:
                            helper.set_mode(TagMode(0))
                    
                    rx.set_mode(TagMode(0))
                    depth = compute_modulation_depth(physics_engine, tags, tx, rx)
                    all_depths.append(depth)
                    
                    if depth > best_depth_for_subset:
                        best_depth_for_subset = depth
                
                subset_best_depths.append(best_depth_for_subset)
                
                # Run heuristic
                print(f"  Running heuristic for subset {subset_names}...")
                heuristic_depth, _ = heuristic_select_helper_config(
                    physics_engine, tags, tx, rx, list(subset), helper_tags
                )
                heuristic_depths.append(heuristic_depth)
                
                if len(subset_list) <= 10:
                    print(f"    Exhaustive={best_depth_for_subset:.2f} mV, "
                          f"Heuristic={heuristic_depth:.2f} mV "
                          f"({heuristic_depth/best_depth_for_subset*100:.1f}%)")

        if all_depths:
            optimal_depth = max(all_depths)
            heuristic_best = max(heuristic_depths) if heuristic_depths else 0.0
            
            all_results.append({
                "helper_count": helper_count,
                "all_depths": all_depths,
                "subset_best_depths": subset_best_depths,
                "heuristic_depths": heuristic_depths,
                "optimal": optimal_depth,
                "heuristic_best": heuristic_best,
            })

    return all_results


def plot_scheme1(results):
    """Plot scheme 1 with exhaustive and heuristic results."""
    plt.figure(figsize=(12, 7))
    
    heuristic_x = []
    heuristic_y = []

    for res in results:
        x = res["helper_count"]
        all_depths = res["all_depths"] 
        subset_bests = res["subset_best_depths"]
        optimal = res["optimal"]
        heuristic_best = res.get("heuristic_best", 0.0)

        # Distribution
        if USE_HISTOGRAM and len(all_depths) > 1:
            hist, bin_edges = np.histogram(all_depths, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_normalized = hist / hist.max() if hist.max() > 0 else hist
            x_offsets = hist_normalized * 0.4
            x_positions = x + x_offsets
            
            plt.plot(x_positions, bin_centers, color='lightgray', linestyle=':', 
                    linewidth=1.5, alpha=0.6, zorder=1)
            plt.plot([x, x_positions[0]], [bin_centers[0], bin_centers[0]], 
                    color='lightgray', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
            plt.plot([x, x_positions[-1]], [bin_centers[-1], bin_centers[-1]], 
                    color='lightgray', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
        elif not USE_HISTOGRAM and len(all_depths) > 0:
            x_jitter = x + (np.random.uniform(-0.15, 0.15, len(all_depths)) if USE_JITTER else 0)
            plt.scatter(x_jitter, all_depths, color='lightgray', alpha=0.3, s=10, zorder=1)

        # Green dots
        if len(subset_bests) > 0:
            x_jitter = x + (np.random.uniform(-0.08, 0.08, len(subset_bests)) if USE_JITTER else 0)
            plt.scatter(x_jitter, subset_bests, color='green', alpha=0.7, s=60, 
                       edgecolors='darkgreen', linewidths=1, zorder=2)

        # Red triangle
        plt.scatter(x, optimal, marker='^', color='red', s=120, 
                   edgecolors='darkred', linewidths=2, zorder=4)

        heuristic_x.append(x)
        heuristic_y.append(heuristic_best)

    # Yellow line
    plt.plot(heuristic_x, heuristic_y, color='gold', linewidth=2.5, 
             marker='o', markersize=8, markerfacecolor='yellow', 
             markeredgecolor='orange', markeredgewidth=2, zorder=3)

    plt.title("Scheme 1: Modulation Depth vs Number of Helpers", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Helpers", fontsize=12)
    plt.ylabel("Modulation Depth (mV)", fontsize=12)
    plt.xticks(range(0, max(heuristic_x) + 1))
    plt.grid(True, alpha=0.3, linestyle='--')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgray', linestyle=':', linewidth=1.5, label='Distribution'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markeredgecolor='darkgreen', markeredgewidth=1, markersize=8, 
               label='Best per subset', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
               markeredgecolor='darkred', markeredgewidth=2, markersize=10, 
               label='Best overall (exhaustive)', linestyle='None'),
        Line2D([0], [0], color='gold', linewidth=2.5, marker='o',
               markersize=8, markerfacecolor='yellow', markeredgecolor='orange',
               label='Heuristic (curve fitting)')
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Heuristic Performance ===")
    for res in results:
        k = res["helper_count"]
        exhaustive = res["optimal"]
        heuristic = res.get("heuristic_best", 0.0)
        ratio = (heuristic / exhaustive * 100) if exhaustive > 0 else 0
        print(f"{k} helpers: Exhaustive={exhaustive:.2f} mV, Heuristic={heuristic:.2f} mV ({ratio:.1f}%)")


if __name__ == "__main__":
    results = run_modulation_depth_sweep()
    plot_scheme1(results)