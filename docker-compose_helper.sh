#!/usr/bin/env bash
# docker-compose_helper.sh — Minimal runner for your current docker-compose.yml
# Services started: mlflow, app
# Notes:
#   - Uses docker-compose v1 syntax.
#   - Loads .env into the environment so your compose file can reference it.
#   - GPU info is shown for awareness only; this script doesn’t toggle GPU/CPU.

set -euo pipefail

echo "==> Loading environment (.env) ..."
if [[ -f .env ]]; then
  # Export all variables defined in .env
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "    ✅ .env loaded."
else
  echo "    ⚠️  .env not found. Proceeding with current shell env."
fi

COMPOSE_FILE_PATH="docker-compose.yml"
if [[ ! -f "$COMPOSE_FILE_PATH" ]]; then
  echo "    ❌ $COMPOSE_FILE_PATH not found in $(pwd)."
  exit 1
fi

echo "==> Environment overview (for MLflow) ..."
echo "    PRIVATEER_MLFLOW_ARTIFACT_PATH = ${PRIVATEER_MLFLOW_ARTIFACT_PATH:-<unset>}"
echo "    PRIVATEER_MLFLOW_DB_PATH       = ${PRIVATEER_MLFLOW_DB_PATH:-<unset>}"
echo "    PRIVATEER_MLFLOW_SERVER_IP     = ${PRIVATEER_MLFLOW_SERVER_IP:-0.0.0.0}"
echo "    PRIVATEER_MLFLOW_SERVER_PORT   = ${PRIVATEER_MLFLOW_SERVER_PORT:-5000}"

echo "==> Starting services from ${COMPOSE_FILE_PATH} ..."
# Tip: with docker-compose v1, unknown keys like 'pull_policy' are ignored gracefully on some setups.
docker-compose -f "$COMPOSE_FILE_PATH" up -d

echo ""
echo "✅ Services are up."
echo "-----------------------------------"
echo "  - MLflow UI: http://localhost:${PRIVATEER_MLFLOW_SERVER_PORT:-5000}"
echo "  - Shell into app: docker exec -it privateer-ad-app /bin/bash"
echo "-----------------------------------"
echo ""
echo "Logs (follow):   docker-compose logs -f"
echo "Stop services:   docker-compose stop"
echo "Tear down:       docker-compose down"
