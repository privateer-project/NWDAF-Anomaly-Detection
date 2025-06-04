import os
import json
import pandas as pd
from fpdf import FPDF
from datetime import datetime

# === Folder Utilities ===

def create_folder_if_not_exists(folder_name):
    """
    Creates a folder if it does not already exist.

    Args:
        folder_name (str): The name or path of the folder to create.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

def create_all_report_folders(base_dir="reports_lime"):
    """
    Creates all subfolders needed to store different formats of LIME reports.

    Args:
        base_dir (str): Base directory for reports.

    Returns:
        dict: A dictionary mapping file types (e.g., 'pdf', 'json') to their folder paths.
    """
    subfolders = ["pdf", "csv", "txt", "json", "html", "xlsx", "md"]
    create_folder_if_not_exists(base_dir)
    for sub in subfolders:
        create_folder_if_not_exists(os.path.join("xAI_lime",base_dir, sub))
    return {ext: os.path.join(base_dir, ext) for ext in subfolders}

# === Report Generation Functions ===

def generate_txt_report_lime(explanation, path_txt):
    """
    Generates a plain text report for LIME explanations.

    Args:
        explanation: The LIME explanation object.
        path_txt (str): Path where the .txt file will be saved.
    """
    with open(path_txt, 'w') as f:
        f.write("=== LIME Explanation Report ===\n\n")
        for feature, weight in explanation.as_list():
            f.write(f"- {feature}: {weight:+.4f}\n")

def generate_csv_report_lime(explanation, path_csv):
    """
    Generates a CSV report with feature weights.

    Args:
        explanation: The LIME explanation object.
        path_csv (str): Path to save the CSV file.
    """
    df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Weight"])
    df.to_csv(path_csv, index=False)

def generate_xlsx_report_lime(explanation, path_xlsx):
    """
    Generates an XLSX (Excel) report with feature weights.

    Args:
        explanation: The LIME explanation object.
        path_xlsx (str): Path to save the XLSX file.
    """
    df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Weight"])
    df.to_excel(path_xlsx, index=False)

def generate_json_report_lime(explanation, path_json, instance_values=None, feature_names=None, model_output=None):
    """
    Generates a JSON report for a LIME explanation, including:
    - the input sample values,
    - the LIME feature weights as a dictionary,
    - a contribution list of features and their weights,
    - the model's predicted output for the instance.

    Args:
        explanation: LIME explanation object.
        path_json (str): Path to save the JSON file.
        instance_values (list): Values of the explained instance.
        feature_names (list): Feature names for the instance.
        model_output (float): Output predicted by the model.
    """
    a=1

    # Convert contributions to list of dicts
    contribution_list = [
        {"feature": feature, "value": float(weight)}
        for feature, weight in explanation.as_list()
    ]

    report = {
        "sample": {name: float(val) for name, val in zip(feature_names, instance_values)} if instance_values and feature_names else {},
        "lime_weights": {feature: float(weight) for feature, weight in explanation.as_list()},
        "contribution": contribution_list,
        "model_output": model_output
    }

    # with open(path_json, 'w') as f:
    #     json.dump(report, f, indent=2)

    return report


def generate_html_report_lime(explanation, path_html):
    """
    Generates an HTML report for visual viewing in a browser.

    Args:
        explanation: The LIME explanation object.
        path_html (str): Path to save the HTML file.
    """
    with open(path_html, 'w') as f:
        f.write("<html><head><title>LIME Report</title></head><body>")
        f.write("<h1>LIME Explanation</h1><ul>")
        for feature, weight in explanation.as_list():
            f.write(f"<li><b>{feature}</b>: {weight:+.4f}</li>")
        f.write("</ul></body></html>")

def generate_pdf_report_lime(explanation, path_pdf):
    """
    Generates a PDF report with feature contributions.

    Args:
        explanation: The LIME explanation object.
        path_pdf (str): Path to save the PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("LIME Report")
    pdf.cell(200, 10, txt="LIME Explanation Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=11)
    for feature, weight in explanation.as_list():
        pdf.cell(200, 8, txt=f"- {feature}: {weight:+.4f}", ln=True)

    pdf.output(path_pdf)

def generate_md_report_lime(explanation, path_md):
    """
    Generates a Markdown report for versioning or documentation.

    Args:
        explanation: The LIME explanation object.
        path_md (str): Path to save the .md file.
    """
    with open(path_md, 'w') as f:
        f.write("# LIME Explanation Report\n\n")
        for feature, weight in explanation.as_list():
            f.write(f"- **{feature}**: {weight:+.4f}\n")

# === Orchestration Function ===

def main_generate_all_lime_reports(data_lime, instance_index=0, base_dir="reports_lime"):
    """
    Main function to generate all LIME report formats for a given explanation.

    Args:
        data_lime (dict): Dictionary containing the LIME explanation and related data.
        instance_index (int): Index of the explained instance.
        base_dir (str): Base folder to save the reports.

    Returns:
        dict: Message indicating success.
    """
    explanation = data_lime["explanation"]
    instance_values = data_lime.get("sample")         # Values for the explained instance
    feature_names = data_lime.get("feature_names")    # Feature names (optional)

    print("Generating reports for LIME explanation...")

    folders = create_all_report_folders(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"instance_{instance_index}_{timestamp}"

    generate_txt_report_lime(explanation, os.path.join(folders["txt"], base_name + ".txt"))
    generate_csv_report_lime(explanation, os.path.join(folders["csv"], base_name + ".csv"))
    generate_xlsx_report_lime(explanation, os.path.join(folders["xlsx"], base_name + ".xlsx"))
    generate_json_report_lime(explanation, os.path.join(folders["json"], base_name + ".json"), instance_values, feature_names)
    generate_html_report_lime(explanation, os.path.join(folders["html"], base_name + ".html"))
    generate_pdf_report_lime(explanation, os.path.join(folders["pdf"], base_name + ".pdf"))
    generate_md_report_lime(explanation, os.path.join(folders["md"], base_name + ".md"))

    print(f"[INFO] All LIME reports for instance {instance_index} generated successfully.")
    return {"message": f"Reports for instance {instance_index} generated successfully."}
