"""
Flask-RESTX routes for management, LIME, and SHAP endpoints.
"""

from typing import Any, Dict

import numpy as np
import torch
import os
from flask import request, send_file, send_from_directory
from flask_restx import Api, Namespace, Resource, fields, reqparse
from werkzeug.utils import secure_filename
from NewContent.privateer_ad.etl.transform import DataProcessor # Consigo carregar o json, mas não consigo carregar o dataset test1.csv
from OldContent.privateer_ad.etl.transform import DataProcessor # Consigo carregar o dataset test1.csv, mas foi preciso alterar um pouco o codigo

from app.config import (
    logger,
    XAI_FEATURE_NAMES,
    XAI_FEATURE_NAMES,
    allowed_dataset_file,
    allowed_model_file,
    MODELS_DIR,
    DATASETS_DIR,
)
from app.xai_app import XAIApplication


def register_routes(api: Api, app, static_folder: str = '../static/browser') -> XAIApplication:
    xai_app = XAIApplication()
    xai_app.load_and_initialize()
    # Cache holders for last computation results
    xai_app.cached_lime_result = None  # type: ignore[attr-defined]
    xai_app.cached_shap_result = None  # type: ignore[attr-defined]

    upload_parser = reqparse.RequestParser()
    upload_parser.add_argument('file', location='files', type='file', required=True)

    xai_input_model = api.model('XAIInput', {
        'data': fields.Raw(required=True, description='Input data for XAI analysis (array of values)'),
        'device': fields.String(required=False, description='Device identifier'),
        'requires_grad': fields.String(required=False, description='Gradient requirement flag'),
        'shape': fields.Raw(required=False, description='Expected data shape')
    })

    management_ns = Namespace('management', description='Model and Dataset Management Operations')
    shap_ns = Namespace('shap', description='SHAP Explanation Operations')
    lime_ns = Namespace('lime', description='LIME Explanation Operations')

    def _send_angular_index():
        # Try multiple common build layouts
        candidate_paths = [
            os.path.join(static_folder, 'index.csr.html'),
            os.path.join(static_folder, 'index.html'),
            os.path.join(static_folder, 'browser', 'index.html'),
            os.path.join(static_folder, 'browser', 'index.csr.html'),
        ]
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return send_file(candidate)
        # If nothing is found, raise a clear error
        raise FileNotFoundError(f"No Angular index found in {static_folder}. Tried: {candidate_paths}")

    @app.route('/')
    def serve_angular_app():
        return _send_angular_index()

    @app.route('/<path:path>')
    def serve_angular_assets(path):
        # 1) Try to serve from static root
        file_path_root = os.path.join(static_folder, path)
        if os.path.exists(file_path_root) and os.path.isfile(file_path_root):
            return send_from_directory(static_folder, path)

        # 2) Try to serve from static/browser subfolder (Angular output often nests here)
        browser_dir = os.path.join(static_folder, 'browser')
        file_path_browser = os.path.join(browser_dir, path)
        if os.path.exists(file_path_browser) and os.path.isfile(file_path_browser):
            # Serve relative to browser_dir
            return send_from_directory(browser_dir, path)

        # 3) If a directory, serve its index.html
        if os.path.isdir(file_path_root):
            dir_index = os.path.join(file_path_root, 'index.html')
            if os.path.exists(dir_index):
                return send_file(dir_index)
        if os.path.isdir(browser_dir):
            dir_path = os.path.join(browser_dir, path)
            if os.path.isdir(dir_path):
                dir_index = os.path.join(dir_path, 'index.html')
                if os.path.exists(dir_index):
                    return send_file(dir_index)

        # 4) Fallback to SPA index
        return _send_angular_index()

    @management_ns.route('/model')
    class ModelManagement(Resource):
        @management_ns.doc(description="Upload a new .pth model file to replace the current model")
        @management_ns.expect(upload_parser)
        def post(self):
            try:
                file = request.files.get('file')
                if not file or file.filename == '':
                    return {"error": "No file provided or filename is empty"}, 400
                if not allowed_model_file(file.filename):
                    return {"error": "Only .pth files are allowed for models"}, 400

                filename = secure_filename(file.filename)
                filepath = os.path.join(MODELS_DIR, filename)

                if os.path.exists(filepath):
                    logger.info(f"Removing existing model file: {filepath}")
                    os.remove(filepath)

                logger.info(f"Saving uploaded model file: {filename}")
                file.save(filepath)

                xai_app.update_model(filepath)

                return {
                    "message": f"Model '{filename}' uploaded and loaded successfully",
                    "filename": filename,
                    "model_path": filepath,
                    "model_parameters": sum(p.numel() for p in xai_app.current_model.parameters())
                }, 200

            except Exception as e:
                logger.error(f"Error uploading/loading model: {str(e)}", exc_info=True)
                return {"error": f"Error uploading/loading model: {str(e)}"}, 500

        @management_ns.doc(description="Get information about the currently loaded model")
        def get(self):
            try:
                if xai_app.current_model is None:
                    return {
                        "model_loaded": False,
                        "model_type": None,
                        "device": None,
                        "parameters": None
                    }, 200

                model_info = {
                    "model_loaded": True,
                    "model_type": type(xai_app.current_model).__name__,
                    "device": str(xai_app.current_model.device) if hasattr(xai_app.current_model, 'device') else "unknown",
                    "parameters": sum(p.numel() for p in xai_app.current_model.parameters()) if hasattr(xai_app.current_model, 'parameters') else 0
                }
                return model_info, 200
            except Exception as e:
                logger.error(f"Error getting model info: {str(e)}", exc_info=True)
                return {"error": f"Error getting model info: {str(e)}"}, 500

    @management_ns.route('/dataset')
    class DatasetManagement(Resource):
        @management_ns.doc(description="Upload a new JSON dataset file to replace the current dataset")
        @management_ns.expect(upload_parser)
        def post(self):
            try:
                file = request.files.get('file')
                if not file or file.filename == '':
                    return {"error": "No file provided or filename is empty"}, 400
                if not allowed_dataset_file(file.filename):
                    return {"error": "Only .json files are allowed for datasets"}, 400

                filename = secure_filename(file.filename)
                filepath = os.path.join(DATASETS_DIR, filename)

                if os.path.exists(filepath):
                    logger.info(f"Removing existing dataset file: {filepath}")
                    os.remove(filepath)

                logger.info(f"Saving uploaded dataset file: {filename}")
                file.save(filepath)

                xai_app.update_dataset(filepath)

                return {
                    "message": f"Dataset '{filename}' uploaded and loaded successfully",
                    "filename": filename,
                    "dataset_path": filepath,
                    "dataset_shape": list(xai_app.current_dataset.shape) if xai_app.current_dataset is not None else None
                }, 200

            except Exception as e:
                logger.error(f"Error uploading/loading dataset: {str(e)}", exc_info=True)
                return {"error": f"Error uploading/loading dataset: {str(e)}"}, 500

        @management_ns.route('/dataset_csv')
        class DatasetCSVManagement(Resource):
            @management_ns.doc(description="Upload a new CSV dataset file to replace the current dataset")
            @management_ns.expect(upload_parser)
            def post(self):
                try:
                    file = request.files.get('file')
                    if not file or file.filename == '':
                        return {"error": "No file provided or filename is empty"}, 400

                    if not file.filename.endswith('.csv'):
                        return {"error": "Only .csv files are allowed for datasets"}, 400

                    filename = secure_filename(file.filename)
                    filepath = os.path.join(DATASETS_DIR, filename)

                    # Salvar com 'with' para garantir fechamento
                    with open(filepath, 'wb') as f:
                        f.write(file.read())

                    # Carregar dataset
                    dl = load_csv(filename)
                    #xai_app.update_dataset(dl)

                    return {
                        "message": f"CSV dataset '{filename}' uploaded and loaded successfully",
                        "filename": filename,
                        "dataset": len(dl),
                        "batch_size": dl.batch_size if hasattr(dl, "batch_size") else None
                    }, 200
                except Exception as e:
                    logger.error(f"Error uploading/loading CSV dataset: {str(e)}", exc_info=True)
                    return {"error": f"Error uploading/loading CSV dataset: {str(e)}"}, 500

        @management_ns.doc(description="Get information about the currently loaded dataset")
        def get(self):
            try:
                if xai_app.current_dataset is None:
                    return {
                        "dataset_loaded": False,
                        "dataset_shape": None,
                        "data_type": None,
                        "device": None
                    }, 200

                dataset_info = {
                    "dataset_loaded": True,
                    "dataset_shape": list(xai_app.current_dataset.shape) if hasattr(xai_app.current_dataset, 'shape') else None,
                    "data_type": str(xai_app.current_dataset.dtype) if hasattr(xai_app.current_dataset, 'dtype') else None,
                    "device": str(xai_app.current_dataset.device) if hasattr(xai_app.current_dataset, 'device') else "unknown"
                }
                return dataset_info, 200
            except Exception as e:
                logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
                return {"error": f"Error getting dataset info: {str(e)}"}, 500

    @lime_ns.route('/lime/calculate')
    class LimeCalculate(Resource):
        @lime_ns.doc(description='Perform LIME explanation analysis for explainable AI', body=xai_input_model,
                     responses={200: 'LIME analysis completed successfully', 400: 'Invalid input', 500: 'Internal error'})
        def post(self):
            try:
                if xai_app.current_model is None or xai_app.current_dataset is None:
                    return {"error": "Model or dataset not loaded. Please upload them first."}, 400

                data = request.get_json()
                if not data:
                    return {"error": "No JSON data provided in request body"}, 400

                instance = XAIApplication.load_tensor_from_json_data(data)
                logger.info("Calculating LIME values for provided data")
                lime_result = xai_app.lime_timeseries.lime_values_from_instance(instance)

                lime_values = lime_result["lime_values"]
                feature_names = lime_result["feature_names"]

                xai_feature_names = XAI_FEATURE_NAMES
                instance_flat = instance.flatten()

                feature_importance: Dict[str, float] = {}
                if lime_values:
                    for feature_idx, importance in lime_values:
                        if feature_idx < len(feature_names):
                            feature_name = feature_names[feature_idx]
                            main_feature = feature_name.rsplit('_', 1)[0] if '_' in feature_name else feature_name
                            if main_feature in xai_feature_names:
                                feature_importance[main_feature] = float(importance)

                for feature in xai_feature_names:
                    if feature not in feature_importance:
                        feature_importance[feature] = 0.0

                json_content = {
                    "message": "LIME explanation calculated successfully.",
                    "sample": {name: float(value) for name, value in zip(xai_feature_names, instance_flat[:8])},
                    "lime_values": feature_importance,
                    "contribution": [{"feature": name, "value": value} for name, value in feature_importance.items()],
                    "feature_names": xai_feature_names,
                    "input_shape": list(instance.shape),
                    "explanation_available": lime_result.get("explanation") is not None
                }
                # Cache the result so it can be retrieved without recomputation
                xai_app.cached_lime_result = json_content  # type: ignore[attr-defined]
                logger.info("LIME values calculated successfully for provided data")
                return json_content, 200
            except Exception as e:
                logger.error(f"Error calculating LIME values: {str(e)}", exc_info=True)
                return {"error": f"Error calculating LIME values: {str(e)}"}, 500

    @lime_ns.route('/lime/last_result')
    class LimeLast(Resource):
        @lime_ns.doc(description='Return the last LIME calculation result without recomputing')
        def get(self):
            try:
                cached = getattr(xai_app, 'cached_lime_result', None)
                if cached is None:
                    return {"error": "No LIME result cached yet. Perform a calculation first."}, 404
                return cached, 200
            except Exception as e:
                logger.error(f"Error returning cached LIME result: {str(e)}", exc_info=True)
                return {"error": f"Error returning cached LIME result: {str(e)}"}, 500

    @shap_ns.route('/shap/calculate')
    class ShapCalculate(Resource):
        @shap_ns.doc(description='Perform SHAP explanation analysis for explainable AI', body=xai_input_model,
                     responses={200: 'SHAP analysis completed successfully', 400: 'Invalid input', 500: 'Internal error'})
        def post(self):
            try:
                if xai_app.current_model is None or xai_app.current_dataset is None:
                    return {"error": "Model or dataset not loaded. Please upload them first."}, 400

                data = request.get_json()
                if not data:
                    return {"error": "No JSON data provided in request body"}, 400

                instance = XAIApplication.load_tensor_from_json_data(data)
                logger.info("Calculating SHAP values for provided data")
                shap_result = xai_app.shap_timeseries.shap_values_from_instance(instance)

                shap_values = shap_result["shap_values"]
                feature_names = XAI_FEATURE_NAMES
                instance_flat = instance.flatten()

                if len(shap_values.shape) > 1:
                    shap_vals_flat = shap_values[0]
                else:
                    shap_vals_flat = shap_values

                def safe_float(value: Any) -> float:
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        return float(value.item())
                    return float(value)

                json_content = {
                    "message": "SHAP values calculated successfully.",
                    "sample": {name: safe_float(value) for name, value in zip(feature_names, instance_flat)},
                    "shap_values": {name: safe_float(val) for name, val in zip(feature_names, shap_vals_flat)},
                    "contribution": [{"feature": name, "value": safe_float(val)} for name, val in zip(feature_names, shap_vals_flat)],
                    "feature_names": feature_names,
                    "input_shape": list(instance.shape)
                }
                # Cache the result so it can be retrieved without recomputation
                xai_app.cached_shap_result = json_content  # type: ignore[attr-defined]
                logger.info("SHAP values calculated successfully for provided data")
                return json_content, 200
            except Exception as e:
                logger.error(f"Error calculating SHAP values: {str(e)}", exc_info=True)
                return {"error": f"Error calculating SHAP values: {str(e)}"}, 500

    @shap_ns.route('/shap/last_result')
    class ShapLast(Resource):
        @shap_ns.doc(description='Return the last SHAP calculation result without recomputing')
        def get(self):
            try:
                cached = getattr(xai_app, 'cached_shap_result', None)
                if cached is None:
                    return {"error": "No SHAP result cached yet. Perform a calculation first."}, 404
                return cached, 200
            except Exception as e:
                logger.error(f"Error returning cached SHAP result: {str(e)}", exc_info=True)
                return {"error": f"Error returning cached SHAP result: {str(e)}"}, 500

    api.add_namespace(management_ns, path='/api')
    api.add_namespace(shap_ns, path='/api')
    api.add_namespace(lime_ns, path='/api')

    return xai_app


#----
# Tentativa de carregar o dataset.csv com o novo script transform.y

def load_csv(filename):
    path = os.path.join(DATASETS_DIR, filename)
    if os.path.exists(path):
        dp = DataProcessor(partition=False)
        data = dp._read_csv(path)
        dataloader  = dp.get_dataloader2(data, seq_len=12, only_benign=False)
        return dataloader
    raise FileNotFoundError


 

