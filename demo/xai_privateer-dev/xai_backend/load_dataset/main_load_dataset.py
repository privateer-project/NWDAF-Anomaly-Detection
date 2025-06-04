import base64
import pickle

from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, reqparse
from werkzeug.utils import secure_filename

# Import helper functions for file operations
from load_dataset.controllers.dataset_contoller import load_csv, list_csv_files, save_csv, delete_csv

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize API documentation with Flask-RESTX
api = Api(app, version='1.0', title='Dataset Management API',
          description='API to manage and load CSV datasets',
          default='Dataset',
          default_label='Dataset Operations')

# Define parser for file upload
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type='file', required=True)

@api.route('/api/datasets/<string:filename>')
class Dataset(Resource):
    @api.doc(description="Uploads a CSV file to the dataset directory.")
    @api.expect(upload_parser)
    def post(self,filename):
        file = request.files.get('file')
        if not file or file.filename == '':
            return {"error": "No file provided or filename is empty"}, 400
        if not file.filename.endswith('.csv'):
            return {"error": "Only .csv files are allowed"}, 400
        try:
            filename = secure_filename(file.filename)
            save_csv(file, filename)
            return {"message": f"File '{filename}' uploaded successfully"}, 200
        except Exception as e:
            return {"error": str(e)}, 500
    
    @api.doc(description="Deletes a specific CSV file.",
             params={'filename': 'Name of the file to delete'},
             responses={
                 200: 'File deleted successfully',
                 404: 'File not found',
                 500: 'Error deleting the file'
             })
    def delete(self, filename):
        try:
            if delete_csv(filename):
                return {"message": f"File '{filename}' deleted successfully"}, 200
            return {"error": f"File '{filename}' not found"}, 404
        except Exception as e:
            return {"error": str(e)}, 500
        
    @api.doc(description="Loads and processes a CSV file, returning the training dataset in base64-encoded format.",
             params={'filename': 'Name of the CSV file to load'},
             responses={
                 200: 'File loaded and processed successfully',
                 404: 'File not found',
                 500: 'Error processing the file'
             })
    def get(self, filename):
        try:
            # Load and process the dataset file into a DataFrame
            dataloader = load_csv(filename)
            print(dataloader)

            # Serialize the Dataset and encode it in base64
            dataset_bytes = base64.b64encode(pickle.dumps(dataloader)).decode("utf-8")
            return {
                "message": f"File '{filename}' loaded and processed successfully",
                "dataset_bytes_base64": dataset_bytes
            }, 200
        except FileNotFoundError as e:
            return {"error": f"File '{filename}' not found: {str(e)}"}, 404
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/datasets/list')
class DatasetList(Resource):
    @api.doc(description="Lists all CSV files available in the dataset directory.")
    def get(self):
        try:
            files = list_csv_files()
            return {"files": files}, 200
        except Exception as e:
            return {"error": str(e)}, 500    

# Optional start method for integration
def start_load_dataset(host, port):
    app.run(host=host, port=port, debug=True)

# Entry point to run the Flask app
if __name__ == '__main__':
    app.run(host="127.0.0.2", port=5000, debug=True)
