"""
Flask-RESTX routes for management, LIME, and SHAP endpoints.
"""

from typing import Any, Dict
import gc
import json
import time
import psutil
import shutil
import tempfile
import glob


import numpy as np
import torch
import os
from flask import request, send_file, send_from_directory
from flask_restx import Api, Namespace, Resource, fields, reqparse
from werkzeug.utils import secure_filename

from app.config import (
    logger,
    XAI_FEATURE_NAMES,
    allowed_dataset_file,
    allowed_model_file,
    MODELS_DIR,
    DATASETS_DIR,
    DATASET_PATH,
    EXAMPLE_INSTANCE
)
from app.xai_engine import XAIApplication

def load_example_data():
    """Load example data from startup_dataset.json for Swagger documentation"""
    try:
        if os.path.exists(EXAMPLE_INSTANCE):
            with open(EXAMPLE_INSTANCE, 'r') as f:
                data = json.load(f)
                return data.get('data', [])
        return []
    except Exception as e:
        logger.warning(f"Could not load example data: {e}")
        return []

def force_remove_file(file_path: str) -> bool:
    """Force remove a file by trying multiple methods including process termination"""
   
    
    # First try normal removal
    try:
        os.remove(file_path)
        logger.info(f"Successfully removed file: {file_path}")
        return True
    except PermissionError:
        logger.warning(f"File is in use, attempting to find and terminate processes: {file_path}")
        
        # Find processes using the file
        processes_using_file = []
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                if proc.info['open_files']:
                    for open_file in proc.info['open_files']:
                        if open_file.path == file_path:
                            processes_using_file.append(proc)
                            logger.info(f"Found process using file: PID={proc.info['pid']}, Name={proc.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Try to terminate processes
        for proc in processes_using_file:
            try:
                logger.info(f"Terminating process PID={proc.info['pid']}")
                proc.terminate()
                proc.wait(timeout=3)  # Wait up to 3 seconds
                logger.info(f"Successfully terminated process PID={proc.info['pid']}")
            except psutil.TimeoutExpired:
                logger.warning(f"Process PID={proc.info['pid']} did not terminate, trying to kill")
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                    logger.info(f"Successfully killed process PID={proc.info['pid']}")
                except psutil.TimeoutExpired:
                    logger.error(f"Could not kill process PID={proc.info['pid']}")
            except Exception as e:
                logger.error(f"Error terminating process PID={proc.info['pid']}: {e}")
        
        # Wait longer for file handles to be released
        time.sleep(2)
        
        # Try multiple removal methods
        removal_methods = [
            lambda: os.remove(file_path),
            lambda: os.unlink(file_path),
            lambda: shutil.rmtree(file_path) if os.path.isdir(file_path) else os.remove(file_path)
        ]
        
        for i, method in enumerate(removal_methods):
            try:
                method()
                logger.info(f"Successfully removed file using method {i+1}: {file_path}")
                return True
            except PermissionError as e:
                logger.warning(f"Method {i+1} failed with PermissionError: {e}")
                continue
            except Exception as e:
                logger.warning(f"Method {i+1} failed with error: {e}")
                continue
        
        # Last resort: try to rename and delete later
        try:
            temp_name = f"{file_path}.deleted_{int(time.time())}"
            os.rename(file_path, temp_name)
            logger.info(f"Renamed file to {temp_name}, will be deleted later")
            return True
        except Exception as e:
            logger.error(f"Could not even rename file: {e}")
            return False
    except Exception as e:
        logger.error(f"Error removing file: {e}")
        return False

def atomic_file_replace(target_path: str) -> bool:
    """Atomically replace a file by creating a temporary file and then swapping"""
    try:
        # Create a temporary file in the same directory
        temp_dir = os.path.dirname(target_path)
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=f".tmp_{int(time.time())}",
            dir=temp_dir,
            prefix=os.path.basename(target_path) + "_"
        )
        
        logger.info(f"Created temporary file: {temp_path}")
        
        # Close the file descriptor so we can work with the file
        os.close(temp_fd)
        
        # The actual file content will be written by the calling function
        # This function just prepares the atomic replacement
        
        # Try to replace the original file
        try:
            # On Windows, we need to remove the target first
            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.move(temp_path, target_path)
            logger.info(f"Successfully atomically replaced: {target_path}")
            return True
        except PermissionError:
            # If we can't replace, try to rename the temp file for later cleanup
            cleanup_name = f"{target_path}.replacement_{int(time.time())}"
            try:
                shutil.move(temp_path, cleanup_name)
                logger.info(f"Renamed temp file to {cleanup_name} for later cleanup")
                return True
            except Exception as e:
                logger.error(f"Could not rename temp file: {e}")
                try:
                    os.remove(temp_path)
                except:
                    pass
                return False
        except Exception as e:
            logger.error(f"Error in atomic replacement: {e}")
            try:
                os.remove(temp_path)
            except:
                pass
            return False
            
    except Exception as e:
        logger.error(f"Error creating temporary file: {e}")
        return False

def cleanup_deleted_files(directory: str) -> None:
    """Clean up files that were renamed for later deletion"""
    
    pattern = os.path.join(directory, "*.deleted_*")
    deleted_files = glob.glob(pattern)
    
    current_time = int(time.time())
    for file_path in deleted_files:
        try:
            # Try to remove files older than 1 minute
            file_age = current_time - int(file_path.split('.deleted_')[-1])
            if file_age > 60:  # 1 minute
                os.remove(file_path)
                logger.info(f"Cleaned up old deleted file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up deleted file {file_path}: {e}")

def remove_existing_variants(base_name: str, ext: str, directory: str) -> None:
    """Remove existing files with the same base name and different suffixes (e.g., _1, _2, etc.)"""
    
    pattern = os.path.join(directory, f"{base_name}_*{ext}")
    existing_files = glob.glob(pattern)
    
    logger.info(f"Looking for variants of {base_name}{ext} in {directory}")
    logger.info(f"Pattern: {pattern}")
    logger.info(f"Found {len(existing_files)} variant files: {existing_files}")
    
    # Clean up old deleted files first
    cleanup_deleted_files(directory)
    
    for file_path in existing_files:
        logger.info(f"Removing existing variant: {file_path}")
        if force_remove_file(file_path):
            logger.info(f"Successfully removed variant: {file_path}")
        else:
            logger.warning(f"Could not remove variant: {file_path}")

def register_routes(api: Api, app, static_folder: str = '../static/browser') -> XAIApplication:
    # Cache holders are already initialized in XAIApplication and populated in load_and_initialize()
    xai_app = XAIApplication()
    xai_app.load_and_initialize()

    upload_parser = reqparse.RequestParser()
    upload_parser.add_argument('file', location='files', type='file', required=True)

    # Load example data from startup_dataset.json
    example_data = load_example_data()

    xai_input_model = api.model('XAIInput', {
        'data': fields.Raw(required=True, description='Input data for XAI analysis (array of values)', example=example_data),
    })

    management_ns = Namespace('management', description='Model and Dataset Management Operations')
    shap_ns = Namespace('shap', description='SHAP Explanation Operations')
    lime_ns = Namespace('lime', description='LIME Explanation Operations')
    classification_ns = Namespace('classification', description='Classification Operations')

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
        @management_ns.doc(description="Upload a new .pt model file to replace the current model")
        @management_ns.expect(upload_parser)
        def post(self):
            try:
                file = request.files.get('file')
                if not file or file.filename == '':
                    return {"error": "No file provided or filename is empty"}, 400
                if not allowed_model_file(file.filename):
                    return {"error": "Only .pt files are allowed for models"}, 400

                filename = secure_filename(file.filename)
                filepath = os.path.join(MODELS_DIR, filename)
                base_name, ext = os.path.splitext(filename)

                # Remove any existing variants of the same file (e.g., model_1.pt, model_2.pt)
                remove_existing_variants(base_name, ext, MODELS_DIR)

                # Handle file replacement with atomic approach
                if os.path.exists(filepath):
                    logger.info(f"File already exists, using atomic replacement: {filepath}")
                    
                    # Force garbage collection to release any file handles
                    gc.collect()
                    
                    # Create temporary file for atomic replacement
                    temp_dir = os.path.dirname(filepath)
                    temp_filename = f"{base_name}_temp_{int(time.time())}{ext}"
                    temp_filepath = os.path.join(temp_dir, temp_filename)
                    
                    logger.info(f"Creating temporary file: {temp_filepath}")
                    
                    # Save the new file to temporary location
                    file.save(temp_filepath)
                    
                    # Try atomic replacement
                    try:
                        # Remove old file and move temp to target
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        os.rename(temp_filepath, filepath)
                        logger.info(f"Successfully replaced file atomically: {filepath}")
                    except PermissionError as e:
                        logger.warning(f"Cannot replace file atomically: {e}")
                        # Keep the temp file and use it as the new file
                        filepath = temp_filepath
                        filename = temp_filename
                        logger.info(f"Using temporary file as replacement: {filepath}")
                else:
                    logger.info(f"File does not exist, saving directly: {filepath}")
                    file.save(filepath)

                # Update the model in XAI engine
                xai_app.update_model(filepath)
                
                # Verify the model was loaded successfully
                if xai_app.current_model is None:
                    return {"error": "Model file uploaded but failed to load in XAI engine"}, 500

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
                        "parameters": None,
                        "model_filename": None
                    }, 200

                model_info = {
                    "model_loaded": True,
                    "model_type": type(xai_app.current_model).__name__,
                    "device": str(xai_app.current_model.device) if hasattr(xai_app.current_model, 'device') else "unknown",
                    "parameters": sum(p.numel() for p in xai_app.current_model.parameters()) if hasattr(xai_app.current_model, 'parameters') else 0,
                    "model_filename": xai_app.current_model_filename
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
                base_name, ext = os.path.splitext(filename)
                
                logger.info(f"Processing upload: original filename='{file.filename}', secure filename='{filename}', target path='{filepath}'")

                # Remove any existing variants of the same file (e.g., dataset_1.json, dataset_2.json)
                remove_existing_variants(base_name, ext, DATASETS_DIR)

                # Handle file replacement with atomic approach
                if os.path.exists(filepath):
                    logger.info(f"File already exists, using atomic replacement: {filepath}")
                    
                    # Force garbage collection to release any file handles
                    gc.collect()
                    
                    # Create temporary file for atomic replacement
                    temp_dir = os.path.dirname(filepath)
                    temp_filename = f"{base_name}_temp_{int(time.time())}{ext}"
                    temp_filepath = os.path.join(temp_dir, temp_filename)
                    
                    logger.info(f"Creating temporary file: {temp_filepath}")
                    
                    # Save the new file to temporary location
                    file.save(temp_filepath)
                    
                    # Try atomic replacement
                    try:
                        # Remove old file and move temp to target
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        os.rename(temp_filepath, filepath)
                        logger.info(f"Successfully replaced file atomically: {filepath}")
                    except PermissionError as e:
                        logger.warning(f"Cannot replace file atomically: {e}")
                        # Keep the temp file and use it as the new file
                        filepath = temp_filepath
                        filename = temp_filename
                        logger.info(f"Using temporary file as replacement: {filepath}")
                else:
                    logger.info(f"File does not exist, saving directly: {filepath}")
                    file.save(filepath)

                # Update the dataset in XAI engine
                xai_app.update_dataset(filepath)
                
                # Verify the dataset was loaded successfully
                if xai_app.current_dataset is None:
                    return {"error": "Dataset file uploaded but failed to load in XAI engine"}, 500

                return {
                    "message": f"Dataset '{filename}' uploaded and loaded successfully",
                    "filename": filename,
                    "dataset_path": filepath,
                    "dataset_shape": list(xai_app.current_dataset.shape) if xai_app.current_dataset is not None else None
                }, 200

            except Exception as e:
                logger.error(f"Error uploading/loading dataset: {str(e)}", exc_info=True)
                return {"error": f"Error uploading/loading dataset: {str(e)}"}, 500

        @management_ns.doc(description="Get information about the currently loaded dataset")
        def get(self):
            try:
                if xai_app.current_dataset is None:
                    return {
                        "dataset_loaded": False,
                        "dataset_shape": None,
                        "data_type": None,
                        "device": None,
                        "dataset_filename": None
                    }, 200

                dataset_info = {
                    "dataset_loaded": True,
                    "dataset_shape": list(xai_app.current_dataset.shape) if hasattr(xai_app.current_dataset, 'shape') else None,
                    "data_type": str(xai_app.current_dataset.dtype) if hasattr(xai_app.current_dataset, 'dtype') else None,
                    "device": str(xai_app.current_dataset.device) if hasattr(xai_app.current_dataset, 'device') else "unknown",
                    "dataset_filename": xai_app.current_dataset_filename
                }
                return dataset_info, 200
            except Exception as e:
                logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
                return {"error": f"Error getting dataset info: {str(e)}"}, 500

    @lime_ns.route('/lime/calculate/string_json')
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

                logger.info("Calculating LIME values for provided data")
                json_content = xai_app.compute_lime_from_json(data)
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

    @shap_ns.route('/shap/calculate/string_json')
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

                logger.info("Calculating SHAP values for provided data")
                json_content = xai_app.compute_shap_from_json(data)
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

    @classification_ns.route('/classification')
    class ClassificationPredict(Resource):
        @classification_ns.doc(description='Return the last classification result (input and output only) without recomputing',
                     responses={200: 'Classification result returned successfully', 404: 'No classification result cached yet'})
        def get(self):
            try:
                cached = getattr(xai_app, 'cached_classification_result', None)
                if cached is None:
                    return {"error": "No classification result cached yet. Perform a LIME or SHAP calculation first."}, 404
                return cached, 200
            except Exception as e:
                logger.error(f"Error returning cached classification result: {str(e)}", exc_info=True)
                return {"error": f"Error returning cached classification result: {str(e)}"}, 500

    api.add_namespace(management_ns, path='/api')
    api.add_namespace(shap_ns, path='/api')
    api.add_namespace(lime_ns, path='/api')
    api.add_namespace(classification_ns, path='/api')

    return xai_app