from flask import Flask, send_from_directory, send_file
from flask_restx import Api, Resource
from flask_cors import CORS
import os
from io import BytesIO
import zipfile
import json
from datetime import datetime
import re

# Import custom SHAP service logic
from xAI_shap.controllers.crud_shap_flask import ShapService
from xAI_shap.controllers.generate_shap_reports import generate_json_report_shap, \
    create_all_report_folders

# === Initialize the Flask app and CORS ===
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# === Configure Flask-RestX API metadata ===
api = Api(app, version='1.0', title='SHAP Management API',
          description='API for managing SHAP explanations.',
          default='SHAP',
          default_label='Operations related to SHAP explanations')

# === Configure file storage ===
UPLOAD_FOLDER = 'graphics'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Instantiate the SHAP service class ===
shap_service = ShapService()

# Define the path to the folder where JSON SHAP reports are stored
JSON_FOLDER = os.path.join("reports_shap", "json")


@api.route('/api_shap/json_report/<int:instance_index>')
class LatestJsonReport(Resource):
    def get(self, instance_index):
        """
        Flask API endpoint that returns the most recent SHAP JSON report for a specific instance.

        Args:
            instance_index (int): The index of the instance for which to retrieve the report.

        Returns:
            dict or tuple: The parsed JSON content of the report, or an error message with HTTP status.
        """
        try:
            # Regular expression to match files named like: instance_<index>_<timestamp>.json
            pattern = re.compile(rf"^instance_{instance_index}_(\d{{8}}_\d{{6}})\.json$")
            matched_files = []

            # Search for matching files in the JSON reports folder
            for filename in os.listdir(JSON_FOLDER):
                match = pattern.match(filename)
                if match:
                    timestamp = match.group(1)
                    matched_files.append((timestamp, filename))

            # If no matching file found, return 404
            if not matched_files:
                return {"error": f"No JSON report found for instance {instance_index}."}, 404

            # Sort files by timestamp in descending order and select the most recent one
            matched_files.sort(reverse=True)
            latest_file = matched_files[0][1]
            full_path = os.path.join(JSON_FOLDER, latest_file)

            # Read and parse the JSON content of the selected file
            with open(full_path, 'r') as f:
                data = json.load(f)

            # Return the parsed JSON content
            return data, 200

        except Exception as e:
            # Return error details in case of failure
            return {"error": str(e)}, 500


# === API endpoint to load dataset and model ===
@api.route('/api_shap/send_data_request/<string:dataset_name>/<string:model_name>')
class InitShap(Resource):
    def get(self, dataset_name, model_name):
        try:
            shap_service.load_resources(dataset_name, model_name)
            return {"message": "Data and model loaded successfully."}, 200
        except Exception as e:
            return {"error": str(e)}, 500


# === API endpoint to run prediction and calculate SHAP values for an instance ===
@api.route('/api_shap/perform_shap_calculations/<int:instance_index>')
class ShapCalculate(Resource):
    def get(self, instance_index):
        try:
            shap_service.instance_index = instance_index
            shap_service.run_predictions()
            shap_service.calculate_shap()

            # Ensure the base directory exists
            base_dir = "reports_shap"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
                print(f"The folder '{base_dir}' has been successfully created.")
            else:
                print(f"The folder '{base_dir}' already exists.")

            # Create necessary subfolders (e.g., reports_shap/json, etc.)
            folders = create_all_report_folders(base_dir=base_dir)

            # Define JSON file path using timestamp for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"instance_{instance_index}_{timestamp}"
            json_path = os.path.join(folders["json"], base_name + ".json")

            # Generate JSON report
            generate_json_report_shap(shap_service.result_shap, json_path, instance_index)

            # Load JSON content to return it in the response
            with open(json_path, 'r') as f:
                json_content = json.load(f)

            return {
                "message": "SHAP values calculated and JSON report generated.",
                "report": json_content
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500

# === API endpoint to generate SHAP explanation graphics ===
@api.route('/api_shap/generation_graphics')
class GenerateGraphics(Resource):
    def get(self):
        try:
            shap_service.generate_graphics()
            return {"message": "Graphics generated."}, 200
        except Exception as e:
            return {"error": str(e)}, 500


# === API endpoint to generate report files ===
@api.route('/generate_reports')
class GenerateReports(Resource):
    def get(self):
        try:
            shap_service.generate_reports()
            return {"message": "Reports generated successfully."}, 200
        except Exception as e:
            return {"error": str(e)}, 500


# === API endpoint to list feature names used for SHAP ===
@api.route('/api_shap/features')
class ListFeatures(Resource):
    def get(self):
        try:
            return {"features": shap_service.list_features()}, 200
        except Exception as e:
            return {"error": str(e)}, 500


# === API endpoint to list generated graphic filenames ===
@api.route('/api_shap/files')
class ListGraphics(Resource):
    def get(self):
        try:
            files = os.listdir(UPLOAD_FOLDER)
            return {"files": files}, 200
        except Exception as e:
            return {"error": str(e)}, 500


# === API endpoint to serve a specific graphic file ===
@api.route('/api_shap/files/<string:filename>')
class ServeGraphic(Resource):
    def get(self, filename):
        try:
            return send_from_directory(UPLOAD_FOLDER, filename)
        except Exception as e:
            return {"error": str(e)}, 500


# === API endpoint to return graphic filenames by instance index ===
@api.route('/api_shap/return_name_graphics/<int:instance_X_test>')
class ReturnGraphicNames(Resource):
    def get(self, instance_X_test):
        try:
            matched_files = [f for f in os.listdir(UPLOAD_FOLDER)
                             if f.startswith(f"shap_summary_{instance_X_test}_") and f.endswith(".png")]
            if not matched_files:
                return {"message": "No graphics found."}, 404
            return {"graphics": matched_files}, 200
        except Exception as e:
            return {"error": str(e)}, 500


# === API endpoint to return graphics as ZIP for a given instance ===
@api.route('/api_shap/return_graphics/<int:instance_X_test>')
class ReturnGraphicsZip(Resource):
    def get(self, instance_X_test):
        try:
            matched_files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)
                             if f.startswith(f"shap_summary_{instance_X_test}_") and f.endswith(".png")]
            if not matched_files:
                return {"message": "No graphics found."}, 404

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for file_path in matched_files:
                    zip_file.write(file_path, os.path.basename(file_path))

            zip_buffer.seek(0)
            return send_file(zip_buffer, as_attachment=True,
                             download_name=f"graphics_instance_{instance_X_test}.zip",
                             mimetype="application/zip")
        except Exception as e:
            return {"error": str(e)}, 500


# === Optional utility function to start the Flask app programmatically ===
def start_xAI_shap(host, port):
    app.run(host=host, port=port, debug=True)


# === Entry point for standalone script execution ===
if __name__ == '__main__':
    app.run(host="127.0.0.4", port=5000, debug=True)
