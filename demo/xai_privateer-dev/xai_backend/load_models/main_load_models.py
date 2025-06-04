import os
import io
import torch
from flask import Flask, request, Response  # Flask for creating the web server and handling requests
from flask_cors import CORS  # For handling Cross-Origin Resource Sharing
from flask_restx import Api, Resource, reqparse  # REST API framework built on Flask
from werkzeug.utils import secure_filename  # Utility to safely handle filenames

# Import the function to load a model from the models directory
from load_models.controllers.model_controller import load_model

# === Constants ===
UPLOAD_FOLDER = 'saved_models'  # Directory to store uploaded model files
ALLOWED_EXTENSIONS = {'pkl'}  # Only allow files with .pkl extension

# Create the model upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Initialize Flask App ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enable CORS for all API routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# === Initialize Flask-RESTX API ===
api = Api(
    app,
    version='1.0',
    title='Model Management API',
    description='API to manage and load PKL model files',
    default='Model',
    default_label='Model Operations'
)

# === Define parser to handle file uploads ===
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type='file', required=True)


# === ROUTE 1: Upload a model file ===
@api.route('/api/models/<string:filename>')
class ModelUpload(Resource):
    @api.doc(description="Uploads a .pt model file to the server.")
    @api.expect(upload_parser)
    def post(self, filename):
        file = request.files.get('file')
        if not file or file.filename == '':
            return {"error": "No file provided or filename is empty"}, 400
        if not file.filename.endswith('.pt'):
            return {"error": "Only .pt files are allowed"}, 400
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Remove existing file with the same name
            if os.path.exists(filepath):
                os.remove(filepath)

            file.save(filepath)
            return {"message": f"File '{filename}' uploaded successfully"}, 200
        except Exception as e:
            return {"error": str(e)}, 500
    
    @api.doc(
        description="Loads a specific .pkl model and returns its state_dict as binary.",
        params={'filename': 'Name of the model file to load'}
    )
    def get(self, filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return {"error": f"File '{filename}' not found"}, 404
        try:
            # Load the model using the helper function
            model = load_model(filename)
            print(model)

            # Serialize the model's state_dict to a binary buffer
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)

            # Return the binary model file in the response
            return Response(buffer.read(), mimetype='application/octet-stream', status=200)
        except Exception as e:
            return {"error": f"Error loading the model: {str(e)}"}, 500
    
    @api.doc(
        description="Deletes a specific .pkl file from the upload directory.",
        params={'filename': 'Name of the file to delete'}
    )
    def delete(self, filename):
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                return {"message": f"File '{filename}' deleted successfully"}, 200
            return {"error": f"File '{filename}' not found"}, 404
        except Exception as e:
            return {"error": str(e)}, 500


# === List all uploaded model files ===
@api.route('/api/models/list')
class ModelList(Resource):
    @api.doc(description="Lists all model files available in the upload directory.")
    def get(self):
        try:
            files = os.listdir(app.config['UPLOAD_FOLDER'])
            return {"files": files if files else []}, 200
        except Exception as e:
            return {"error": str(e)}, 500    


# === Optional utility function to start the app programmatically ===
def start_load_model(host, port):
    app.run(host=host, port=port, debug=True)


# === Entry point to run the app manually ===
if __name__ == '__main__':
    app.run(host="127.0.0.3", port=5000, debug=True)
