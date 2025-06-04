import os
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to allow saving plots in headless environments (e.g., servers)
import matplotlib.pyplot as plt
import pandas as pd

def plot_summary(shap_values, x, feature_names, instance_index, plot_type="bar"):
    """
    Generates a SHAP summary plot for a specific instance.
    Supported types: 'bar', 'dot', 'violin'.
    """
    print(f"Generating Summary Plot ({plot_type}) for instance {instance_index}...")

    # Extract SHAP values and corresponding feature values for the given instance
    shap_val_raw = np.array(shap_values[instance_index])
    x_val_raw = np.array(x[instance_index]).flatten()

    shap_values_single = shap_val_raw.reshape(1, -1)
    x_single = x_val_raw.reshape(1, -1)

    print(f"[DEBUG] shap_values_single.shape: {shap_values_single.shape}")
    print(f"[DEBUG] x_single.shape: {x_single.shape}")
    print(f"[DEBUG] len(feature_names): {len(feature_names)}")

    # Plot and save to file
    plt.figure()
    shap.summary_plot(
        shap_values_single,
        x_single,
        feature_names=feature_names,
        plot_type=plot_type,
        show=False
    )
    summary_path = os.path.join("xAI_shap","graphics", f"shap_summary_{plot_type}_instance_{instance_index}.png")
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    return summary_path

def plot_beeswarm(shap_values, x, feature_names, instance_index):
    """
    Generates a dot-based beeswarm SHAP summary plot.
    """
    return plot_summary(shap_values, x, feature_names, instance_index, plot_type="dot")

def plot_dependence(shap_values, x, feature_names, feature_index=0, instance_index=0):
    """
    Generates a dependence plot for the selected feature across all instances.
    """
    if not feature_names or len(shap_values) < 2:
        print("Dependence plot requires multiple instances. Skipping...")
        return None

    print(f"Generating Dependence Plot for feature '{feature_names[feature_index]}'...")

    try:
        # Flatten the input arrays for SHAP and features
        x_2d = np.array([xi.flatten() for xi in x])
        shap_vals_2d = np.array([np.array(sv) for sv in shap_values])

        # Create the dependence plot
        plt.figure()
        shap.dependence_plot(
            feature_names[feature_index],
            shap_vals_2d,
            pd.DataFrame(x_2d, columns=feature_names),
            feature_names=feature_names,
            interaction_index=None,
            show=False
        )
        dependence_path = os.path.join("xAI_shap","graphics", f"shap_dependence_instance_{instance_index}_{feature_names[feature_index]}.png")
        plt.savefig(dependence_path, bbox_inches="tight")
        plt.close()
        return dependence_path
    except Exception as e:
        print(f"[ERROR] Failed to generate dependence plot: {e}")
        return None

def plot_force(explainer, shap_values, x, feature_names, instance_index):
    """
    Generates a force plot for a specific instance, saved as an HTML file.
    """
    print(f"Generating Force Plot for instance {instance_index}...")

    try:
        expected_value = explainer.expected_value
        if not hasattr(expected_value, "__len__"):
            expected_value = [expected_value]

        force_plot = shap.force_plot(
            expected_value[0],
            np.array(shap_values[instance_index]),
            np.array(x[instance_index]).flatten(),
            feature_names=feature_names
        )

        force_path = os.path.join("xAI_shap","graphics", f"shap_force_instance_{instance_index}.html")
        shap.save_html(force_path, force_plot)
        return force_path
    except Exception as e:
        print(f"[ERROR] Failed to generate force plot: {e}")
        return None

def plot_waterfall(explainer, shap_values, feature_names, instance_index):
    """
    Generates a waterfall plot showing additive SHAP contributions for a single instance.
    """
    print(f"Generating Waterfall Plot for instance {instance_index}...")

    try:
        expected_value = explainer.expected_value
        if not hasattr(expected_value, "__len__"):
            expected_value = [expected_value]

        plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            expected_value[0],
            np.array(shap_values[instance_index]),
            feature_names=feature_names,
            show=False
        )
        waterfall_path = os.path.join("xAI_shap","graphics", f"shap_waterfall_instance_{instance_index}.png")
        plt.savefig(waterfall_path, bbox_inches="tight")
        plt.close()
        return waterfall_path
    except Exception as e:
        print(f"[ERROR] Failed to generate waterfall plot: {e}")
        return None

def plot_decision(explainer, shap_values, instance_index, feature_names):
    """
    Generates a decision plot showing cumulative SHAP impact.
    """
    print(f"Generating Decision Plot for instance {instance_index}...")
    try:
        plt.figure()
        shap.decision_plot(
            explainer.expected_value,
            np.array(shap_values),
            feature_names=feature_names,
            show=False
        )
        path = os.path.join("xAI_shap","graphics", f"shap_decision_instance_{instance_index}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path
    except Exception as e:
        print(f"[ERROR] Failed to generate decision plot: {e}")
        return None

def plot_heatmap(shap_values, feature_names):
    """
    Generates a heatmap of SHAP values for all instances and features.
    """
    print("Generating Heatmap for all instances...")
    try:
        plt.figure()
        shap_values_2d = np.array(shap_values)
        shap.plots.heatmap(shap_values_2d, feature_names=feature_names, show=False)
        path = os.path.join("xAI_shap","graphics", "shap_heatmap_all_instances.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path
    except Exception as e:
        print(f"[ERROR] Failed to generate heatmap: {e}")
        return None

def main_graphics_shap(results, instance):
    """
    Main function that orchestrates the generation of all SHAP plots
    for a specified instance. Also includes a heatmap for all instances.

    Args:
        results (dict): Contains SHAP values, feature names, input samples, and the explainer.
        instance (int): Index of the instance to visualize.
    """
    shap_values = results["shap_values"]
    x = results["x"]
    explainer = results["explainer"]
    feature_names = results["feature_names"]

    print("shap_values type:", type(shap_values))
    print("shap_values shape (raw):", np.array(shap_values).shape)
    print("x shape:", np.array(x).shape)

    shap_values = np.array(shap_values)
    if shap_values.ndim == 1:
        shap_values = np.tile(shap_values, (len(x), 1))
    elif shap_values.shape[0] != len(x):
        raise ValueError(f"shap_values shape {shap_values.shape} does not match number of instances {len(x)}")

    os.makedirs("graphics", exist_ok=True)
    plots_by_instance = {}

    # Only generate plots for the requested instance
    i = instance
    print(f"\nProcessing SHAP plots for instance {i}...")
    plots = {
        "summary_bar": plot_summary(shap_values, x, feature_names, i, plot_type="bar"),
        "summary_dot": plot_beeswarm(shap_values, x, feature_names, i),
        "dependence_plot": plot_dependence(shap_values, x, feature_names, feature_index=0, instance_index=i),
        "force_plot": plot_force(explainer, shap_values, x, feature_names, i),
        "waterfall_plot": plot_waterfall(explainer, shap_values, feature_names, i),
        "decision_plot": plot_decision(explainer, shap_values, i, feature_names)
    }
    plots_by_instance[i] = plots

    # Add heatmap across all instances
    plots_by_instance["heatmap"] = plot_heatmap(shap_values, feature_names)

    print("\nAll SHAP plots have been generated successfully!")
    return plots_by_instance
