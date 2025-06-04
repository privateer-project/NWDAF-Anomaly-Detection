import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for safe multiprocessing rendering


def plot_lime_bar(explanation, output_dir, instance_index, mode):
    """
    Generates a horizontal bar chart showing the top LIME feature contributions.

    Args:
        explanation: The LIME explanation object.
        output_dir (str): Directory where the plot image will be saved.
        instance_index (int): Index of the explained instance.
        mode (str): Model type ("regression" or "classification").

    Returns:
        str: Path to the saved bar chart image.
    """
    print(f"Generating Summary Bar Plot for instance {instance_index} ({mode})...")

    explanation_list = explanation.as_list()
    if not explanation_list:
        print("No explanation data to generate the bar chart.")
        return None

    df = pd.DataFrame(explanation_list, columns=["Feature", "Weight"])
    df = df.sort_values(by="Weight", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(df["Feature"], df["Weight"], color="skyblue")
    plt.xlabel("Contribution")
    plt.title(f"Top Features - LIME Explanation ({mode}) (Instance {instance_index})")
    plt.tight_layout()

    bar_path = os.path.join(output_dir, f"lime_bar_{mode}_instance_{instance_index}.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    print("Bar chart saved at:", bar_path)

    return bar_path

def plot_lime_table(explanation, output_dir, instance_index, mode):
    """
    Generates a table plot summarizing LIME feature contributions.

    Args:
        explanation: The LIME explanation object.
        output_dir (str): Directory where the table image will be saved.
        instance_index (int): Index of the explained instance.
        mode (str): Model type ("regression" or "classification").

    Returns:
        str: Path to the saved table image.
    """
    print(f"Generating Table for instance {instance_index} ({mode})...")

    explanation_list = explanation.as_list()
    if not explanation_list:
        print("No explanation data to generate the table.")
        return None

    df = pd.DataFrame(explanation_list, columns=["Feature", "Weight"])

    plt.figure(figsize=(8, 2 + 0.5 * len(df)))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    table_path = os.path.join(output_dir, f"lime_table_{mode}_instance_{instance_index}.png")
    plt.savefig(table_path, bbox_inches="tight")
    plt.close()
    print("Table saved at:", table_path)

    return table_path

def plot_lime_html(explanation, output_dir, instance_index, mode):
    """
    Exports the LIME explanation as an interactive HTML file.

    Args:
        explanation: The LIME explanation object.
        output_dir (str): Directory where the HTML file will be saved.
        instance_index (int): Index of the explained instance.
        mode (str): Model type ("regression" or "classification").

    Returns:
        str: Path to the saved HTML file.
    """
    html_path = os.path.join(output_dir, f"lime_explanation_{mode}_instance_{instance_index}.html")
    explanation.save_to_file(html_path)
    print("HTML explanation saved at:", html_path)
    return html_path

def main_graphics_lime(explanation_data, instance_index, mode="regression"):
    """
    Main function that generates all graphical outputs for a given LIME explanation.

    Args:
        explanation_data (dict): Result returned from `main_calculation_lime`.
        instance_index (int): Index of the instance being explained.
        mode (str): Type of model explanation ("regression" or "classification").

    Returns:
        dict: Dictionary containing paths to all generated visualizations.
    """
    output_dir = os.path.join("xAI_lime","graphics")
    os.makedirs(output_dir, exist_ok=True)

    explanation = explanation_data["explanation"]

    print(f"\n[INFO] Generating LIME plots for instance {instance_index} ({mode})...")
    plots = {
        "bar": plot_lime_bar(explanation, output_dir, instance_index, mode),
        "table": plot_lime_table(explanation, output_dir, instance_index, mode),
        "html": plot_lime_html(explanation, output_dir, instance_index, mode)
    }

    print(f"[INFO] All LIME plots for instance {instance_index} generated successfully.")
    return {instance_index: plots}
