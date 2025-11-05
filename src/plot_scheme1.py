import itertools
import numpy as np
import matplotlib.pyplot as plt

from tags.state_machine import StateSerializer
from tags.tag import TagMode
from main import load_json
from physics import PhysicsEngine
from state import AppState


# ---------------- CONFIG ----------------
CONFIG_PATH = "demo/7_tag_test/config_files/config.json"
NUM_COEFFS = 6  # Only use reflection coefficients starting from index 1 (indices 1 - NUM_COEFFS) based on available; index 0 is non-reflecting

# Optional plotting settings
USE_JITTER = True  # Set to False for perfectly vertical alignment of dots
USE_HISTOGRAM = True  # Set to False to show individual gray dots instead of histogram
# ----------------------------------------


def mW_to_mV(v):
    """Helper: convert volts to millivolts."""
    return v * 1000.0


def compute_modulation_depth(physics_engine, tags, tx, rx):
    """
    Compute modulation depth with current tag configuration.
    Returns modulation depth (mV).
    """
    # Two modes of TX1 for modulation depth calculation
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


def run_modulation_depth_sweep():
    app_state = AppState()
    serializer = StateSerializer()

    # Load config and initialize physics engine
    exciters, tags, _, _ = load_json(CONFIG_PATH, serializer, app_state=app_state)
    physics_engine = PhysicsEngine(exciters)
    tx = tags.get("TX1")
    rx = tags.get("RX1")
    helper_tags = [tag for name, tag in tags.items() if name not in ("TX1", "RX1")]
    
    print(f"Number of helpers: {len(helper_tags)}")
    print(f"TX chip impedances: {len(tx.chip_impedances)}")
    print(f"Testing with coefficient indices 1-{NUM_COEFFS} (index 0 = non-reflecting)")

    all_results = []
    subsets_by_count = generate_helper_subsets(helper_tags, max_helpers=min(5, len(helper_tags)))

    for helper_count, subset_list in subsets_by_count.items():
        print(f"\n=== Evaluating with {helper_count} helpers ===")
        print(f"Number of subsets: {len(subset_list)}")

        all_depths = []  # All modulation depths for ALL combinations
        subset_best_depths = []  # Best depth for each subset
        
        for subset_idx, subset in enumerate(subset_list):
            subset_names = [tag.get_name() for tag in subset]
            best_depth_for_subset = 0.0
            
            if helper_count == 0:
                # Baseline: no helpers active, all in non-reflecting mode
                for helper in helper_tags:
                    helper.set_mode(TagMode(0))  # non-reflecting
                
                rx.set_mode(TagMode(0))
                
                depth = compute_modulation_depth(physics_engine, tags, tx, rx)
                all_depths.append(depth)
                best_depth_for_subset = depth
                subset_best_depths.append(depth)
                
                print(f"Baseline (no helpers): {depth:.2f} mV")
            else:
                # Test all reflection coefficient combinations for this subset
                # Use indices 1 through NUM_COEFFS (not 0, which is non-reflecting)
                coeff_indices = range(1, NUM_COEFFS + 1)
                num_combinations = len(coeff_indices) ** helper_count
                
                combo_count = 0
                for combo in itertools.product(coeff_indices, repeat=helper_count):
                    # Set up helpers and RX coefficients for this subset
                    for helper, coeff_idx in zip(subset, combo):
                        helper.set_mode(TagMode(coeff_idx))
                    
                    for helper in helper_tags:
                        if helper not in subset:
                            helper.set_mode(TagMode(0))
                    
                    rx.set_mode(TagMode(0))
                    
                    # Compute modulation depth
                    depth = compute_modulation_depth(physics_engine, tags, tx, rx)
                    all_depths.append(depth)
                    
                    if depth > best_depth_for_subset:
                        best_depth_for_subset = depth
                    
                    combo_count += 1
                
                subset_best_depths.append(best_depth_for_subset)
                
                # Progress update
                if len(subset_list) > 10 and (subset_idx + 1) % max(1, len(subset_list) // 5) == 0:
                    print(f"  Progress: {subset_idx + 1}/{len(subset_list)} subsets, "
                          f"best so far: {best_depth_for_subset:.2f} mV")
                elif len(subset_list) <= 10:
                    print(f"  Subset {subset_names}: best = {best_depth_for_subset:.2f} mV "
                          f"({num_combinations} combinations tested)")

        if all_depths:
            optimal_depth = max(all_depths)
            
            all_results.append({
                "helper_count": helper_count,
                "all_depths": all_depths,
                "subset_best_depths": subset_best_depths,
                "optimal": optimal_depth,
            })
            
            print(f"  Summary for {helper_count} helpers:")
            print(f"    Total measurements: {len(all_depths)}")
            print(f"    Min: {min(all_depths):.2f} mV")
            print(f"    Max: {optimal_depth:.2f} mV")
            print(f"    Mean: {np.mean(all_depths):.2f} mV")
            print(f"    Std: {np.std(all_depths):.2f} mV")

    return all_results


def plot_scheme1(results):
    """
    Plot a scheme 1 modulation depth chart.
    
    - Gray: distribution shown as histogram (USE_HISTOGRAM=True) or all dots (USE_HISTOGRAM=False)
    - Green dots: best modulation depth for each subset (one per subset)
    - Red triangle: globally best configuration (one per helper count)
    """
    plt.figure(figsize=(12, 7))
    
    global_max_y = -1
    global_max_x = -1

    for res in results:
        x = res["helper_count"]
        all_depths = res["all_depths"] 
        subset_bests = res["subset_best_depths"]
        optimal = res["optimal"]

        if USE_HISTOGRAM:
            if len(all_depths) > 1:
                hist, bin_edges = np.histogram(all_depths, bins=50)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Normalize histogram to get density (0 to 1)
                hist_normalized = hist / hist.max() if hist.max() > 0 else hist
                
                # Scale x-offset by density (0 to 0.4 range)
                x_offsets = hist_normalized * 0.4
                x_positions = x + x_offsets
                
                plt.plot(x_positions, bin_centers, color='gray', linestyle=':', 
                        linewidth=1.5, alpha=0.6, zorder=1)
                
                plt.plot([x, x_positions[0]], [bin_centers[0], bin_centers[0]], 
                        color='gray', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
                plt.plot([x, x_positions[-1]], [bin_centers[-1], bin_centers[-1]], 
                        color='gray', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
        else:
            if len(all_depths) > 0:
                if USE_JITTER:
                    x_jitter = x + np.random.uniform(-0.15, 0.15, len(all_depths))
                else:
                    x_jitter = np.full(len(all_depths), x)
                
                plt.scatter(x_jitter, all_depths, color='lightgray', alpha=0.3, s=10, zorder=1)

        if len(subset_bests) > 0:
            if USE_JITTER:
                x_jitter = x + np.random.uniform(-0.08, 0.08, len(subset_bests))
            else:
                x_jitter = np.full(len(subset_bests), x)
            
            plt.scatter(x_jitter, subset_bests, color='green', alpha=0.7, s=60, 
                       edgecolors='darkgreen', linewidths=1, zorder=2)

        plt.scatter(x, optimal, marker='^', color='red', s=120, 
                   edgecolors='darkred', linewidths=2, zorder=4)

        if optimal > global_max_y:
            global_max_y = optimal
            global_max_x = x

    plt.title("Scheme 1: Modulation Depth vs Number of Helpers", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Helpers", fontsize=12)
    plt.ylabel("Modulation Depth (mV)", fontsize=12)
    
    helper_counts = [r["helper_count"] for r in results]
    plt.xticks(range(0, max(helper_counts) + 1))
    plt.grid(True, alpha=0.3, linestyle='--')
    
    from matplotlib.lines import Line2D
    if USE_HISTOGRAM:
        legend_elements = [
            Line2D([0], [0], color='lightgray', linestyle=':', linewidth=1.5,
                   label='Distribution (density)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markeredgecolor='darkgreen', markeredgewidth=1, markersize=8, 
                   label='Best per subset', linestyle='None'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=2, markersize=10, 
                   label='Best overall (exhaustive)', linestyle='None'),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                   markersize=6, label='All measurements', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markeredgecolor='darkgreen', markeredgewidth=1, markersize=8, 
                   label='Best per subset', linestyle='None'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=2, markersize=10, 
                   label='Best overall (exhaustive)', linestyle='None'),
        ]
    
    plt.legend(handles=legend_elements, loc="upper left", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Global Best Configuration ===")
    print(f"Helper count: {global_max_x}")
    print(f"Modulation depth: {global_max_y:.2f} mV")
    
    print(f"\n=== Improvement Trend ===")
    baseline = results[0]["optimal"] if len(results) > 0 else 0
    for res in results:
        k = res["helper_count"]
        improvement = ((res["optimal"] - baseline) / baseline * 100) if baseline > 0 else 0
        num_subsets = len(res["subset_best_depths"])
        num_measurements = len(res["all_depths"])
        print(f"{k} helpers: {res['optimal']:.2f} mV ({improvement:+.1f}% vs baseline) - "
              f"{num_subsets} subsets, {num_measurements} total measurements")


if __name__ == "__main__":
    results = run_modulation_depth_sweep()
    plot_scheme1(results)