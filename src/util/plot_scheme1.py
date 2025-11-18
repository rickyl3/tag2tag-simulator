import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cmath

# ---------------- CONFIG ----------------
CSV_DATA_PATH = "src/util/data.csv"
NUM_COEFFS = 10
USE_JITTER = True
USE_HISTOGRAM = True
NUM_TRAINING_SAMPLES = 100
NUM_TOP_PREDICTIONS = 5
USE_CSV_DATA = True
COMPARE_HEURISTICS = True
RECEIVER_TAG_ID = 4  # Which tag ID is the receiver (1,2,3,4,5,7,10)
# ----------------------------------------


def get_gamma_set():
    """Get the reflection coefficients matching chip_impedances config."""
    gammas = [0 + 0j]  # Index 0: non-reflecting
    # Indices 1-10 corresponding to chip_impedances
    for theta in [60, 30, 0, -30, -60, 30, -30, 0, 45, -45]:
        gammas.append(cmath.exp(1j * np.deg2rad(theta)))
    return gammas


def model_modulation_depth(gamma_config, *params):
    """Simple model without feedback loops."""
    gammas = get_gamma_set()
    
    tx_gamma_0_idx = int(gamma_config[0])
    tx_gamma_1_idx = int(gamma_config[1])
    helper_gamma_indices = [int(g) for g in gamma_config[2:]]
    
    A_e = params[0]
    h_er = complex(params[1], params[2])
    h_et = complex(params[3], params[4])
    h_tr = complex(params[5], params[6])
    
    signal_0 = A_e * h_er + A_e * h_et * gammas[tx_gamma_0_idx] * h_tr
    signal_1 = A_e * h_er + A_e * h_et * gammas[tx_gamma_1_idx] * h_tr
    
    param_idx = 7
    for helper_gamma_idx in helper_gamma_indices:
        if helper_gamma_idx != 0:
            h_eh = complex(params[param_idx], params[param_idx + 1])
            h_hr = complex(params[param_idx + 2], params[param_idx + 3])
            contribution = A_e * h_eh * gammas[helper_gamma_idx] * h_hr
            signal_0 += contribution
            signal_1 += contribution
        param_idx += 4
    
    return abs(abs(signal_1) - abs(signal_0))


def model_modulation_depth_feedback(gamma_config, *params):
    """Model with feedback loops: S = (I - HΓ)^(-1) h_exciter"""
    gammas = get_gamma_set()
    
    tx_gamma_0_idx = int(gamma_config[0])
    tx_gamma_1_idx = int(gamma_config[1])
    helper_gamma_indices = [int(g) for g in gamma_config[2:]]
    num_helpers = len(helper_gamma_indices)
    
    # Total tags: TX + RX + helpers
    n = 2 + num_helpers
    A_e = params[0]
    param_idx = 1
    
    # Extract H matrix (n x n, skip diagonal)
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                H[i, j] = complex(params[param_idx], params[param_idx + 1])
                param_idx += 2
    
    # Extract exciter vector
    h_exciter = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        h_exciter[i] = A_e * complex(params[param_idx], params[param_idx + 1])
        param_idx += 2
    
    # Compute for TX at gamma_0
    gammas_0 = np.zeros(n, dtype=np.complex128)
    gammas_0[0] = gammas[tx_gamma_0_idx]  # TX
    gammas_0[1] = 0  # RX non-reflecting
    for i, g_idx in enumerate(helper_gamma_indices):
        gammas_0[2 + i] = gammas[g_idx]
    
    Gamma_0 = np.diag(gammas_0)
    I = np.eye(n, dtype=np.complex128)
    try:
        S_0 = np.linalg.solve(I - H @ Gamma_0, h_exciter)
        signal_0 = abs(S_0[1])  # RX is at index 1
    except:
        signal_0 = 0
    
    # Compute for TX at gamma_1
    gammas_1 = gammas_0.copy()
    gammas_1[0] = gammas[tx_gamma_1_idx]
    Gamma_1 = np.diag(gammas_1)
    try:
        S_1 = np.linalg.solve(I - H @ Gamma_1, h_exciter)
        signal_1 = abs(S_1[1])
    except:
        signal_1 = 0
    
    return abs(signal_1 - signal_0)


def fit_model_parameters(training_data, num_helpers, use_feedback=False):
    """Fit model parameters using training data."""
    if len(training_data) < 20:
        return None
    
    X_data = np.array([config for config, _ in training_data])
    y_data = np.array([depth for _, depth in training_data])
    
    if use_feedback:
        n = 2 + num_helpers
        num_h_elements = n * (n - 1) * 2
        num_exciter_elements = n * 2
        num_params = 1 + num_h_elements + num_exciter_elements
        
        p0 = np.concatenate([
            [np.mean(y_data) * 0.01],
            np.random.randn(num_params - 1) * 0.05
        ])
        model_func = model_modulation_depth_feedback
    else:
        num_params = 7 + num_helpers * 4
        p0 = np.concatenate([
            [np.mean(y_data) * 0.01],
            np.random.randn(num_params - 1) * 0.1
        ])
        model_func = model_modulation_depth
    
    try:
        def model_wrapper(X, *params):
            return np.array([model_func(x, *params) for x in X])
        
        bounds_lower = [0.001] + [-10] * (num_params - 1)
        bounds_upper = [10] + [10] * (num_params - 1)
        
        popt, _ = curve_fit(
            model_wrapper, X_data, y_data, p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=3000, ftol=1e-4, xtol=1e-4
        )
        return popt
    except Exception as e:
        print(f"    Curve fitting failed ({('feedback' if use_feedback else 'simple')}): {e}")
        return None


def load_csv_complete_dataset(csv_path, receiver_tag_id):
    """Load and filter CSV by receiver tag."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
    except FileNotFoundError:
        print(f"Warning: CSV file {csv_path} not found.")
        return None
    
    df_filtered = df[df['Receiver'] == receiver_tag_id].copy()
    print(f"Filtered to {len(df_filtered)} rows for Receiver={receiver_tag_id}")
    
    if len(df_filtered) == 0:
        print(f"Warning: No data found for Receiver={receiver_tag_id}")
        return None
    
    return df_filtered


def get_helper_tag_columns(receiver_tag_id):
    """Get list of CSV column names for helper tags (excluding receiver)."""
    all_tags = {1: "Tag1", 2: "Tag2", 3: "Tag3", 4: "Tag4", 
                5: "Tag5", 7: "Tag7", 10: "Tag10"}
    
    helper_cols = []
    for tag_id, col_name in all_tags.items():
        if tag_id != receiver_tag_id:
            helper_cols.append(col_name)
    
    return helper_cols


def heuristic_select_helper_config(csv_df, helper_cols, helper_subset_cols, 
                                   use_feedback=False, num_training=NUM_TRAINING_SAMPLES):
    """
    Use curve fitting to select best helper configuration.
    
    Args:
        csv_df: Filtered dataframe (already filtered by receiver)
        helper_cols: All possible helper column names
        helper_subset_cols: Helper columns in current subset
        use_feedback: Use feedback model
        num_training: Training samples
    
    Returns:
        (best_depth, helper_subset_cols, best_phase_config)
    """
    if len(helper_subset_cols) == 0:
        # For 0 helpers, find where all helpers are at phase 5
        zero_helper_df = csv_df.copy()
        for col in helper_cols:
            if col in zero_helper_df.columns:
                zero_helper_df = zero_helper_df[zero_helper_df[col] == 5]
        
        if len(zero_helper_df) > 0:
            best_row = zero_helper_df.loc[zero_helper_df['Median Readings'].idxmax()]
            return float(best_row['Median Readings']), [], []
        return 0.0, [], []
    
    # Filter to rows where:
    # - Helpers in subset have phase != 5 
    # - Helpers not in subset have phase == 5
    filtered_df = csv_df.copy()
    
    for col in helper_cols:
        if col in filtered_df.columns:
            if col in helper_subset_cols:
                filtered_df = filtered_df[filtered_df[col] != 5]
            else:
                filtered_df = filtered_df[filtered_df[col] == 5]
    
    if len(filtered_df) == 0:
        print(f"    No matching data for subset {helper_subset_cols}")
        return 0.0, helper_subset_cols, None
    
    print(f"    Found {len(filtered_df)} configurations for subset")
    
    # Collect training data
    training_data = []
    sample_size = min(num_training, len(filtered_df))
    sample_rows = filtered_df.sample(n=sample_size, random_state=42)
    
    for _, row in sample_rows.iterrows():
        tx_gamma_0, tx_gamma_1 = 1, 2  # TX modulates between these
        helper_gammas = tuple(int(row[col]) for col in helper_subset_cols)
        
        measured_depth = float(row['Median Readings'])
        gamma_config = (tx_gamma_0, tx_gamma_1) + helper_gammas
        training_data.append((gamma_config, measured_depth))
    
    # Fit model
    num_helpers = len(helper_subset_cols)
    fitted_params = fit_model_parameters(training_data, num_helpers, use_feedback=use_feedback)
    
    if fitted_params is None:
        best_row = filtered_df.loc[filtered_df['Median Readings'].idxmax()]
        best_config = tuple(int(best_row[col]) for col in helper_subset_cols)
        return float(best_row['Median Readings']), helper_subset_cols, best_config
    
    # Predict all configurations
    predictions = []
    model_func = model_modulation_depth_feedback if use_feedback else model_modulation_depth
    
    for _, row in filtered_df.iterrows():
        helper_gammas = tuple(int(row[col]) for col in helper_subset_cols)
        gamma_config = (1, 2) + helper_gammas
        try:
            predicted_depth = model_func(gamma_config, *fitted_params)
        except:
            predicted_depth = 0.0
        predictions.append((helper_gammas, predicted_depth, float(row['Median Readings'])))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Return best actual from top predictions
    best_depth = 0.0
    best_config = None
    for config, pred, actual in predictions[:NUM_TOP_PREDICTIONS]:
        if actual > best_depth:
            best_depth = actual
            best_config = config
    
    return best_depth, helper_subset_cols, best_config


def run_modulation_depth_sweep():
    """Main sweep function."""
    print(f"Receiver Tag ID: {RECEIVER_TAG_ID}")
    print(f"Testing coefficient indices 1-{NUM_COEFFS}")

    # Load CSV
    csv_df = load_csv_complete_dataset(CSV_DATA_PATH, RECEIVER_TAG_ID)
    if csv_df is None:
        print("Error: Could not load CSV data")
        return []

    # Get helper columns (exclude receiver)
    helper_cols = get_helper_tag_columns(RECEIVER_TAG_ID)
    print(f"Helper columns: {helper_cols}")

    all_results = []
    
    # Generate subsets
    max_helpers = min(5, len(helper_cols))
    subsets_by_count = {}
    for k in range(max_helpers + 1):
        subsets_by_count[k] = list(itertools.combinations(helper_cols, k))

    for helper_count, subset_list in subsets_by_count.items():
        print(f"\n=== Evaluating with {helper_count} helpers ===")

        all_depths = []
        subset_best_depths = []
        heuristic_simple_depths = []
        heuristic_feedback_depths = []
        heuristic_simple_subsets = []
        heuristic_feedback_subsets = []
        
        # Get all depths for this helper count
        filtered_df = csv_df.copy()
        
        # Count active helpers (phase != 5)
        active_cols = []
        for col in helper_cols:
            if col in filtered_df.columns:
                filtered_df[f'{col}_active'] = (filtered_df[col] != 5).astype(int)
                active_cols.append(f'{col}_active')
        
        if active_cols:
            filtered_df['active_helpers'] = filtered_df[active_cols].sum(axis=1)
            filtered_df = filtered_df[filtered_df['active_helpers'] == helper_count]
        
        all_depths = filtered_df['Median Readings'].tolist()
        print(f"  Found {len(all_depths)} configurations with {helper_count} helpers")
        
        # Evaluate each subset
        for subset_cols in subset_list:
            # Get best for this subset
            subset_df = csv_df.copy()
            for col in helper_cols:
                if col in subset_df.columns:
                    if col in subset_cols:
                        subset_df = subset_df[subset_df[col] != 5]
                    else:
                        subset_df = subset_df[subset_df[col] == 5]
            
            if len(subset_df) > 0:
                best_for_subset = subset_df['Median Readings'].max()
                subset_best_depths.append(best_for_subset)
            else:
                subset_best_depths.append(0.0)
                best_for_subset = 0.0
            
            # Run heuristics
            if helper_count > 0:
                if len(subset_list) <= 20:
                    print(f"  Running heuristics for {subset_cols}...")
                
                # Simple heuristic
                depth_simple, _, _ = heuristic_select_helper_config(
                    csv_df, helper_cols, list(subset_cols), use_feedback=False
                )
                heuristic_simple_depths.append(depth_simple)
                heuristic_simple_subsets.append(subset_cols)
                
                # Feedback heuristic
                if COMPARE_HEURISTICS:
                    depth_feedback, _, _ = heuristic_select_helper_config(
                        csv_df, helper_cols, list(subset_cols), use_feedback=True
                    )
                    heuristic_feedback_depths.append(depth_feedback)
                    heuristic_feedback_subsets.append(subset_cols)
                
                if len(subset_list) <= 20:
                    print(f"    Best={best_for_subset:.2f}, Simple={depth_simple:.2f}" +
                          (f", Feedback={depth_feedback:.2f}" if COMPARE_HEURISTICS else ""))
            else:
                heuristic_simple_depths.append(best_for_subset)
                heuristic_simple_subsets.append([])
                if COMPARE_HEURISTICS:
                    heuristic_feedback_depths.append(best_for_subset)
                    heuristic_feedback_subsets.append([])

        if all_depths:
            # Find best heuristic selections
            heuristic_simple_best = max(heuristic_simple_depths) if heuristic_simple_depths else 0.0
            heuristic_simple_best_idx = heuristic_simple_depths.index(heuristic_simple_best) if heuristic_simple_depths else 0
            heuristic_simple_best_subset = heuristic_simple_subsets[heuristic_simple_best_idx] if heuristic_simple_subsets else []
            
            if COMPARE_HEURISTICS and heuristic_feedback_depths:
                heuristic_feedback_best = max(heuristic_feedback_depths)
            else:
                heuristic_feedback_best = 0.0
            
            # Find exhaustive best for simple heuristic's subset
            exhaustive_best_for_subset = 0.0
            if heuristic_simple_best_subset:
                subset_df = csv_df.copy()
                for col in helper_cols:
                    if col in subset_df.columns:
                        if col in heuristic_simple_best_subset:
                            subset_df = subset_df[subset_df[col] != 5]
                        else:
                            subset_df = subset_df[subset_df[col] == 5]
                
                if len(subset_df) > 0:
                    exhaustive_best_for_subset = subset_df['Median Readings'].max()
            elif helper_count == 0:
                exhaustive_best_for_subset = heuristic_simple_best
            
            all_results.append({
                "helper_count": helper_count,
                "all_depths": all_depths,
                "subset_best_depths": subset_best_depths,
                "heuristic_simple_depths": heuristic_simple_depths,
                "heuristic_feedback_depths": heuristic_feedback_depths,
                "optimal": max(all_depths),
                "heuristic_simple_best": heuristic_simple_best,
                "heuristic_feedback_best": heuristic_feedback_best,
                "exhaustive_best_for_simple_subset": exhaustive_best_for_subset,
            })

    return all_results


def plot_scheme1(results):
    """Plot results."""
    plt.figure(figsize=(14, 8))
    
    heuristic_simple_x, heuristic_simple_y = [], []
    heuristic_feedback_x, heuristic_feedback_y = [], []
    exhaustive_for_subset_x, exhaustive_for_subset_y = [], []

    for res in results:
        x = res["helper_count"]
        all_depths = res["all_depths"] 
        subset_bests = res["subset_best_depths"]
        heuristic_simple_best = res.get("heuristic_simple_best", 0.0)
        heuristic_feedback_best = res.get("heuristic_feedback_best", 0.0)
        exhaustive_best_for_subset = res.get("exhaustive_best_for_simple_subset", 0.0)

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

        # Green dots
        if len(subset_bests) > 0:
            x_jitter = x + (np.random.uniform(-0.08, 0.08, len(subset_bests)) if USE_JITTER else 0)
            plt.scatter(x_jitter, subset_bests, color='green', alpha=0.7, s=60, 
                       edgecolors='darkgreen', linewidths=1, zorder=2)

        # Red triangle
        plt.scatter(x, exhaustive_best_for_subset, marker='^', color='red', s=120, 
                   edgecolors='darkred', linewidths=2, zorder=4)

        heuristic_simple_x.append(x)
        heuristic_simple_y.append(heuristic_simple_best)
        exhaustive_for_subset_x.append(x)
        exhaustive_for_subset_y.append(exhaustive_best_for_subset)
        
        if COMPARE_HEURISTICS:
            heuristic_feedback_x.append(x)
            heuristic_feedback_y.append(heuristic_feedback_best)

    # Yellow line
    plt.plot(heuristic_simple_x, heuristic_simple_y, color='gold', linewidth=2.5, 
             marker='o', markersize=8, markerfacecolor='yellow', 
             markeredgecolor='orange', markeredgewidth=2, zorder=3)

    # Blue line
    if COMPARE_HEURISTICS and heuristic_feedback_y:
        plt.plot(heuristic_feedback_x, heuristic_feedback_y, color='deepskyblue', linewidth=2.5, 
                 marker='s', markersize=8, markerfacecolor='cyan', 
                 markeredgecolor='blue', markeredgewidth=2, zorder=3)

    plt.title(f"Modulation Depth vs Helpers (Receiver=Tag{RECEIVER_TAG_ID})" + 
              (" - Comparing Heuristics" if COMPARE_HEURISTICS else ""), 
              fontsize=14, fontweight='bold')
    plt.xlabel("Number of Helpers", fontsize=12)
    plt.ylabel("Modulation Depth (mV)", fontsize=12)
    plt.xticks(range(0, max(heuristic_simple_x) + 1))
    plt.grid(True, alpha=0.3, linestyle='--')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgray', linestyle=':', linewidth=1.5, 
               label='Distribution (CSV)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markeredgecolor='darkgreen', markeredgewidth=1, markersize=8, 
               label='Best per subset', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
               markeredgecolor='darkred', markeredgewidth=2, markersize=10, 
               label='Best phase for heuristic subset', linestyle='None'),
        Line2D([0], [0], color='gold', linewidth=2.5, marker='o',
               markersize=8, markerfacecolor='yellow', markeredgecolor='orange',
               label='Simple heuristic')
    ]
    
    if COMPARE_HEURISTICS:
        legend_elements.append(
            Line2D([0], [0], color='deepskyblue', linewidth=2.5, marker='s',
                   markersize=8, markerfacecolor='cyan', markeredgecolor='blue',
                   label='Feedback heuristic')
        )
    
    plt.legend(handles=legend_elements, loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Heuristic Performance ===")
    for res in results:
        k = res["helper_count"]
        exhaustive = res["exhaustive_best_for_simple_subset"]
        simple = res.get("heuristic_simple_best", 0.0)
        feedback = res.get("heuristic_feedback_best", 0.0)
        
        simple_ratio = (simple / exhaustive * 100) if exhaustive > 0 else 0
        feedback_ratio = (feedback / exhaustive * 100) if exhaustive > 0 else 0
        
        print(f"{k} helpers: Exhaustive={exhaustive:.2f} mV, "
              f"Simple={simple:.2f} mV ({simple_ratio:.1f}%)" +
              (f", Feedback={feedback:.2f} mV ({feedback_ratio:.1f}%)" if COMPARE_HEURISTICS else ""))


if __name__ == "__main__":
    results = run_modulation_depth_sweep()
    plot_scheme1(results)