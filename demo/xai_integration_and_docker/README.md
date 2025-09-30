XAI Time-Series Application
===========================

Overview
--------
This repository contains a full-stack Explainable AI (XAI) application for time-series anomaly detection. The backend exposes REST APIs to compute LIME and SHAP explanations over a Transformer-based PyTorch model, while the frontend (Angular) provides interactive views for results and diagnostics. The Angular application is built to static assets and served by the Flask backend.

Key Features
------------
- PyTorch Transformer model loading and inference
- SHAP and LIME explanations for time-series inputs
- Organized, modular backend: clear separation of config, routes and app services
- Angular SPA frontend, served as static files by Flask
- Production-ready Docker setup with multi-stage build and health checks

Architecture
------------
- Backend (Flask): `backend/`
  - Entry point: `backend/api_xAI.py`
  - App modules: `backend/app/`
    - `config.py` – logging, constants, paths, helpers
    - `xai_app.py` – model/dataset lifecycle and XAI services
    - `routes.py` – REST routes and Angular static serving
  - XAI utilities: `backend/shap_timeseries.py`, `backend/lime_timeseries.py` (if present)
  - Model and dataset mounts: `backend/models`, `backend/datasets`
- Frontend (Angular): `frontend/`
  - Source code, Angular config and build tooling
  - Build output is configured to project-level `static/`
- Static assets: `static/`
  - Contains Angular build artifacts (e.g., `static/browser/*`)

Directory Layout
----------------
- `backend/` – Flask app, XAI logic, API endpoints
- `frontend/` – Angular application (development sources)
- `static/` – Angular production build output (served by Flask)
- `Dockerfile` – multi-stage build (Angular → Flask runtime)
- `docker-compose.yml` – local composition with bind mounts and healthcheck

Local Development
-----------------
1) Backend (virtualenv recommended)
```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python api_xAI.py
```
Backend will start on http://127.0.0.1:5000. Swagger UI: http://127.0.0.1:5000/swagger

2) Frontend (optional during development)
```bash
cd frontend
npm ci
npm start  # or: ng serve
```
When ready for production build:
```bash
cd frontend
npm run build
```
Build artifacts will be placed under `../static`.

Docker (Build and Run)
----------------------
1) Build image
```bash
docker compose build --no-cache
```

2) Start container
```bash
docker compose up -d
```

3) Logs and health
```bash
docker compose logs -f xai-app
# Healthcheck endpoint (served by Flask-RESTX):
# http://localhost:5000/swagger.json
```

4) Access
- API docs: http://localhost:5000/swagger
- Frontend SPA: http://localhost:5000/home

Environment and Configuration
-----------------------------
- FLASK_ENV: defaults to `production` in container
- PYTHONPATH: set to `/app` in container
- Paths (inside container):
  - Angular static: `/app/static`
  - Models: `/app/backend/models`
  - Datasets: `/app/backend/datasets`
These are bind-mounted via `docker-compose.yml` for local development.

API Namespaces (High-level)
---------------------------
- Management: `/api/model`, `/api/dataset` (upload, info)
- LIME: `/api/lime/calculate`
- SHAP: `/api/shap/calculate`
See Swagger UI for request/response schemas.

Troubleshooting
---------------
- Blank pages after build: ensure Flask serves the correct static folder (`static/`) and that Angular assets exist under `static/browser`. Clear browser cache (Ctrl+F5).
- 404s on API calls from frontend: align frontend URLs to `http://127.0.0.1:5000/api/...` or configure a proxy appropriately.
- Model loading failures: verify the model path and contents under `backend/models` (or the configured path).

License
-------
This project is provided as-is; please add your license terms here.

