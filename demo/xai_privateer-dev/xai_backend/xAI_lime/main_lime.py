import datetime
import os
import json
import re
import zipfile
from io import BytesIO

from flask import Flask, send_from_directory, send_file, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS

from xAI_lime.controllers.crud_lime_flask import LimeService
from xAI_lime.controllers.generate_lime_reports import (
    create_all_report_folders,
    generate_json_report_lime,
    main_generate_all_lime_reports
)

# === Initialize Flask application and API ===
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow cross-origin requests for API routes

api = Api(app, version="1.0", title="LIME Management API",
          description="API to manage LIME explanations and report generation",
          default="LIME",
          default_label="LIME Operations")

# === Directory configuration ===
UPLOAD_FOLDER = 'graphics'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

JSON_FOLDER = os.path.join("reports_lime", "json")
lime_service = LimeService()


# === API Endpoints ===

@api.route('/api_lime/send_data_request/<string:name_dataset>/<string:name_model>')
class InitLime(Resource):
    def get(self, name_dataset, name_model):
        """
        Load dataset and model resources using external services.
        """
        try:
            lime_service.load_resources(name_dataset, name_model)
            return {"message": "Data and model loaded successfully."}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api_lime/perform_lime_calculations/<int:instance_X_test>')
class LimeCalculate(Resource):
    def get(self, instance_X_test):
        """
        Perform LIME explanation for a specific instance and save the result as a JSON report.
        """
        try:

            lime_service.instance_index = instance_X_test
            lime_service.run_predictions()
            lime_service.calculate_lime()

            # Prepare directory and filename for the report
            base_dir = "reports_lime"
            folders = create_all_report_folders(base_dir=base_dir)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"instance_{instance_X_test}_{timestamp}"
            json_path = os.path.join(folders["json"], base_name + ".json")

            # Extract explanation and instance data
            explanation = lime_service.result_lime["explanation"]
            instance_values = lime_service.result_lime.get("sample")
            feature_names = lime_service.result_lime.get("feature_names")

            # Generate the JSON report
            json_content = generate_json_report_lime(
                explanation,
                json_path,
                instance_values=instance_values,
                feature_names=feature_names,
                model_output=lime_service.result_lime.get("model_output")
            )

            # with open(json_path, 'r') as f:
            #     json_content = json.load(f)

            return {
                "message": "LIME explanation calculated and JSON report generated.",
                "report": json_content
            }, 200

        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api_lime/json_report/<int:instance_index>')
class LatestJsonReport(Resource):
    def get(self, instance_index):
        """
        Return the most recent JSON report for the given instance.
        """
        try:
            pattern = re.compile(rf"^instance_{instance_index}_(\d{{8}}_\d{{6}})\.json$")
            matched_files = []

            for filename in os.listdir(JSON_FOLDER):
                match = pattern.match(filename)
                if match:
                    timestamp = match.group(1)
                    matched_files.append((timestamp, filename))

            if not matched_files:
                return {"error": f"No JSON report found for instance {instance_index}."}, 404

            matched_files.sort(reverse=True)
            latest_file = matched_files[0][1]
            full_path = os.path.join(JSON_FOLDER, latest_file)

            with open(full_path, 'r') as f:
                data = json.load(f)

            return data, 200

        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api_lime/generation_graphics')
class GenerateGraphics(Resource):
    def get(self):
        """
        Generate visual explanation plots (bar chart, table, HTML) for the current instance.
        """
        try:
            lime_service.generate_graphics()
            return {"message": "Graphics generated."}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api_lime/features')
class ListFeatures(Resource):
    def get(self):
        """
        Return the list of feature names used in the model.
        """
        try:
            return {"features": lime_service.list_features()}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api_lime/files')
class ListGraphics(Resource):
    def get(self):
        """
        List all image files in the graphics output directory.
        """
        try:
            files = os.listdir(UPLOAD_FOLDER)
            return {"files": files}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api_lime/files/<string:filename>')
class ServeGraphic(Resource):
    def get(self, filename):
        """
        Serve a single image file (PNG) from the graphics directory.
        """
        try:
            return send_from_directory(UPLOAD_FOLDER, filename)
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api_lime/return_name_graphics/<int:instance_X_test>')
class ReturnGraphicNames(Resource):
    def get(self, instance_X_test):
        """
        List all graphics generated for the given instance.
        """
        try:
            matched_files = [f for f in os.listdir(UPLOAD_FOLDER)
                             if f.startswith(f"lime_summary_{instance_X_test}_") and f.endswith(".png")]
            if not matched_files:
                return {"message": "No graphics found."}, 404
            return {"graphics": matched_files}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api_lime/return_graphics/<int:instance_X_test>')
class ReturnGraphicsZip(Resource):
    def get(self, instance_X_test):
        """
        Package all graphics for a given instance into a ZIP file and send as download.
        """
        try:
            matched_files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)
                             if f.startswith(f"lime_summary_{instance_X_test}_") and f.endswith(".png")]
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


@api.route('/generate_reports')
class GenerateReports(Resource):
    def get(self):
        """
        Generate all report formats (TXT, CSV, PDF, HTML, etc.) for the current instance.
        """
        try:
            instance_index = lime_service.instance_index
            data_lime = lime_service.result_lime
            result = main_generate_all_lime_reports(data_lime, instance_index)
            return result, 200
        except Exception as e:
            return {"error": str(e)}, 500

# Define input model including instance_X_test
lime_input_model = api.model('LIMEInput', {
    'device': fields.String(required=True, description='Test instance identifier'),
    'data': fields.Raw(required=True, description='Input data for SHAP analysis'),
    'requires_grad': fields.String(required=True, description='Test instance identifier'),
    'shape': fields.Raw(required=True, description='Input data for SHAP analysis')
})

# === API endpoint to return graphics as ZIP for a given instance ===
@api.route('/api/lime/perform_XAI/<string:dataset_name>/<string:model_name>')
class LiveLIME(Resource):
    """
    Perform LIME explainability analysis on a given test instance.
    
    This endpoint computes SHAP values to explain model predictions,
    providing insights into feature contributions for the specified instance.
     """
    @api.doc('perform_shap_analysis',
             description='Perform SHAP (SHapley Additive exPlanations) analysis for explainable AI',
             params={
                 'dataset_name': 'Name of the dataset to use for SHAP analysis',
                 'model_name': 'Name of the model to use for SHAP analysis'
             },
             body=lime_input_model,
             responses={
                #  200: ('SHAP analysis completed successfully', shap_output_model),
                 200: 'SHAP analysis completed successfully',
                 400: 'Invalid input data or parameters',
                 422: 'Unable to process the provided data',
                 500: 'Internal server error during SHAP computation'
             })
    def post(self,dataset_name, model_name):
        data = request.get_json()
        #data = request.stream.read()
        tensor = lime_service.load_tensor_from_json_data(data)
        if tensor is None:
            return {"error": "Invalid input data format."}, 400
        lime_service.load_resources("test1.csv", "model.pt")
        lime_service.run_predictions()
        lime_service.calculate_lime_from_tensor(tensor)
        json_path = "temp2.json"
        # Generate JSON report
        generate_json_report_lime(lime_service.result_lime['explanation'], json_path, 0)

        # Load JSON content to return it in the response
        with open(json_path, 'r') as f:
            json_content = json.load(f)

        return {
            "message": "SHAP values calculated and JSON report generated.",
            "report": json_content
        }, 200
        # return data['data'], 200


# === Optional method to run the Flask app programmatically ===
def start_xAI_lime(host, port):
    app.run(host=host, port=port, debug=True)


# === Run the application directly if this script is executed ===
if __name__ == '__main__':
    app.run(host="127.0.0.5", port=5000, debug=True)
