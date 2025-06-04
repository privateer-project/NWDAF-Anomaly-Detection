import os
import json
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import numpy as np
import torch

# === Utility Functions ===

def safe_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        else:
            return float(value.view(-1)[0].item())
    elif isinstance(value, (np.ndarray, list)):
        return float(value[0])
    else:
        return float(value)

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        print(f"The folder '{folder_name}' has been successfully created.")
    else:
        print(f"The folder '{folder_name}' already exists.")

def create_all_report_folders(base_dir="reports_shap"):
    subfolders = ["pdf", "csv", "txt", "json", "html", "xlsx", "md"]
    create_folder_if_not_exists(base_dir)
    for sub in subfolders:
        create_folder_if_not_exists(os.path.join("xAI_shap",base_dir, sub))
    return {ext: os.path.join(base_dir, ext) for ext in subfolders}

def _prepare_instance_vals(data, instance_index):
    instance_vals = data["x"][instance_index]
    if isinstance(instance_vals, torch.Tensor):
        instance_vals = instance_vals.detach().cpu().numpy()
    return instance_vals.flatten()

# === Report Generation Functions ===

def generate_txt_report_shap(data, path_txt, instance_index=0):
    shap_vals = data["shap_values"]
    if isinstance(shap_vals, (np.ndarray, torch.Tensor)) and shap_vals.ndim > 1:
        shap_vals = shap_vals[instance_index]
    instance_vals = _prepare_instance_vals(data, instance_index)

    with open(path_txt, 'w') as f:
        f.write("=== SHAP Explanation Report ===\n\n")
        f.write("Explained Instance:\n")
        for name, value in zip(data["feature_names"], instance_vals):
            f.write(f"  - {name}: {safe_float(value):.4f}\n")
        f.write("\nFeature SHAP Values:\n")
        for name, shap_val in zip(data["feature_names"], shap_vals):
            f.write(f"  - {name}: {safe_float(shap_val):.4f}\n")

def generate_csv_report_shap(data, path_csv, instance_index=0):
    shap_vals = data["shap_values"]
    if isinstance(shap_vals, (np.ndarray, torch.Tensor)) and shap_vals.ndim > 1:
        shap_vals = shap_vals[instance_index]
    instance_vals = _prepare_instance_vals(data, instance_index)

    df = pd.DataFrame({
        "Feature": data["feature_names"],
        "Value": [safe_float(val) for val in instance_vals],
        "SHAP Value": [safe_float(val) for val in shap_vals]
    })
    df.to_csv(path_csv, index=False)

def generate_xlsx_report_shap(data, path_xlsx, instance_index=0):
    shap_vals = data["shap_values"]
    if isinstance(shap_vals, (np.ndarray, torch.Tensor)) and shap_vals.ndim > 1:
        shap_vals = shap_vals[instance_index]
    instance_vals = _prepare_instance_vals(data, instance_index)

    df = pd.DataFrame({
        "Feature": data["feature_names"],
        "Value": [safe_float(val) for val in instance_vals],
        "SHAP Value": [safe_float(val) for val in shap_vals]
    })
    df.to_excel(path_xlsx, index=False)

def generate_json_report_shap(data, path_json, instance_index=0):
    shap_vals = data["shap_values"]
    if isinstance(shap_vals, (np.ndarray, torch.Tensor)) and shap_vals.ndim > 1:
        shap_vals = shap_vals[instance_index]
    instance_vals = _prepare_instance_vals(data, instance_index)

    report = {
        "sample": {name: safe_float(value) for name, value in zip(data["feature_names"], instance_vals)},
        "shap_values": {name: safe_float(val) for name, val in zip(data["feature_names"], shap_vals)},
        "contribution": [
            {"feature": name, "value": safe_float(val)}
            for name, val in zip(data["feature_names"], shap_vals)
        ],
        "model_output": data.get("model_output")  # Adiciona isto no main_calculation_shap
    }

    # with open(path_json, 'w') as f:
    #     json.dump(report, f, indent=2)
    
    return report

def generate_html_report_shap(data, path_html, instance_index=0):
    shap_vals = data["shap_values"]
    if isinstance(shap_vals, (np.ndarray, torch.Tensor)) and shap_vals.ndim > 1:
        shap_vals = shap_vals[instance_index]
    instance_vals = _prepare_instance_vals(data, instance_index)

    with open(path_html, 'w') as f:
        f.write("<html><head><title>SHAP Report</title></head><body>")
        f.write("<h1>SHAP Explanation Report</h1>")
        f.write("<h2>Explained Instance</h2><ul>")
        for name, value in zip(data["feature_names"], instance_vals):
            f.write(f"<li><b>{name}</b>: {safe_float(value):.4f}</li>")
        f.write("</ul><h2>SHAP Values</h2><ul>")
        for name, shap_val in zip(data["feature_names"], shap_vals):
            f.write(f"<li><b>{name}</b>: {safe_float(shap_val):.4f}</li>")
        f.write("</ul></body></html>")

def generate_pdf_report_shap(data, path_pdf, instance_index=0):
    shap_vals = data["shap_values"]
    if isinstance(shap_vals, (np.ndarray, torch.Tensor)) and shap_vals.ndim > 1:
        shap_vals = shap_vals[instance_index]
    instance_vals = _prepare_instance_vals(data, instance_index)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("SHAP Report")
    pdf.cell(200, 10, txt="SHAP Explanation Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)

    pdf.cell(200, 10, txt="Explained Instance:", ln=True)
    for name, value in zip(data["feature_names"], instance_vals):
        pdf.cell(200, 8, txt=f" - {name}: {safe_float(value):.4f}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="SHAP Values:", ln=True)
    for name, shap_val in zip(data["feature_names"], shap_vals):
        pdf.cell(200, 8, txt=f" - {name}: {safe_float(shap_val):.4f}", ln=True)

    pdf.output(path_pdf)

def generate_md_report_shap(data, path_md, instance_index=0):
    shap_vals = data["shap_values"]
    if isinstance(shap_vals, (np.ndarray, torch.Tensor)) and shap_vals.ndim > 1:
        shap_vals = shap_vals[instance_index]
    instance_vals = _prepare_instance_vals(data, instance_index)

    with open(path_md, 'w') as f:
        f.write("# SHAP Explanation Report\n\n")
        f.write("## Explained Instance\n")
        for name, value in zip(data["feature_names"], instance_vals):
            f.write(f"- **{name}**: {safe_float(value):.4f}\n")
        f.write("\n## SHAP Values\n")
        for name, shap_val in zip(data["feature_names"], shap_vals):
            f.write(f"- **{name}**: {safe_float(shap_val):.4f}\n")

# === Main Orchestration Function ===

def main_generate_all_shap_reports(data_shap, instance_index=0, base_dir="reports_shap"):
    shap_vals = data_shap["shap_values"]
    print("Feature names:", len(data_shap["feature_names"]))
    print("Type of shap_vals[instance_index]:", type(shap_vals[instance_index]))

    folders = create_all_report_folders(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"instance_{instance_index}_{timestamp}"

    generate_txt_report_shap(data_shap, os.path.join(folders["txt"], base_name + ".txt"), instance_index)
    generate_csv_report_shap(data_shap, os.path.join(folders["csv"], base_name + ".csv"), instance_index)
    generate_xlsx_report_shap(data_shap, os.path.join(folders["xlsx"], base_name + ".xlsx"), instance_index)
    generate_json_report_shap(data_shap, os.path.join(folders["json"], base_name + ".json"), instance_index)
    generate_html_report_shap(data_shap, os.path.join(folders["html"], base_name + ".html"), instance_index)
    generate_pdf_report_shap(data_shap, os.path.join(folders["pdf"], base_name + ".pdf"), instance_index)
    generate_md_report_shap(data_shap, os.path.join(folders["md"], base_name + ".md"), instance_index)

    print(f"[INFO] All SHAP reports for instance {instance_index} were generated successfully.")
    return {"message": f"Reports for instance {instance_index} generated successfully."}
