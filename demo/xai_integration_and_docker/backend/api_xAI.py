"""
Slim entrypoint for the XAI API server.

This module constructs the Flask app, applies CORS, builds the Flask-RESTX Api,
and registers all routes via app.routes.register_routes. Model/dataset handling
and route implementations live in the app/ package.
"""

from flask import Flask
from flask_cors import CORS
from flask_restx import Api

from app.config import logger
from app.routes import register_routes

# Initialize Flask application and API
app = Flask(__name__, static_folder='../static', static_url_path='/static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

api = Api(
    app,
    version="1.0",
    title="XAI Management API",
          description="API to manage LIME and SHAP explanations with organized namespaces",
          default="XAI",
          default_label="XAI Operations",
    doc="/swagger",
)

# Register routes and initialize XAI services
xai_app = register_routes(api, app, static_folder=app.static_folder)


def start_xai_api(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Start the XAI API server."""
    app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
    start_xai_api()