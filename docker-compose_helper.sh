#!/usr/bin/env bash
# docker-compose_helper.sh — Minimal runner for your current docker-compose.yml
# Services started:
#   - default: mlflow, app
#   - with --alveo (or --alveo=true): mlflow, app-fpga
#
# Notes:
#   - Uses docker-compose v1 syntax.
#   - Loads .env into the environment so your compose file can reference it.
#   - Prints Alveo-related device info for awareness; Compose runs app-fpga with --privileged.

set -euo pipefail

# --- Parse args ---
ALVEO=false
usage() {
  cat <<EOF
Usage: $(basename "$0") [--alveo[=true|false]] [--help]

  --alveo           Start mlflow + app-fpga instead of mlflow + app.
                    If provided without value, it defaults to true.
  --alveo=true      Same as --alveo
  --alveo=false     Disable app-fpga; start mlflow + app (default)
  --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --alveo) ALVEO=true; shift ;;
    --alveo=*)
      val="${1#*=}"; case "${val,,}" in
        true|1|yes|on)  ALVEO=true  ;;
        false|0|no|off) ALVEO=false ;;
        *) echo "Invalid value for --alveo: $val"; usage; exit 2 ;;
      esac; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

echo "==> Loading environment (.env) ..."
if [[ -f .env ]]; then
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

# Informational device scan if Alveo mode
if [[ "$ALVEO" == "true" ]]; then
  echo "==> Alveo mode requested (--alveo=true) ..."
  echo "    Scanning for FPGA/Xilinx related devices (informational) ..."
  xcl=$(find /dev -maxdepth 1 -name "xclmgmt*" 2>/dev/null || true)
  rend=$(find /dev/dri -maxdepth 1 -name "renderD*" 2>/dev/null || true)
  if [[ -n "${xcl}${rend}" ]]; then
    echo "    Found devices:"
    [[ -n "$xcl"  ]] && echo "      - ${xcl//$'\n'/$'\n        '}"
    [[ -n "$rend" ]] && echo "      - ${rend//$'\n'/$'\n        '}"
  else
    echo "    (No xclmgmt*/renderD* devices detected; continuing anyway with --privileged)"
  fi
fi

SERVICES=("mlflow")
if [[ "$ALVEO" == "true" ]]; then
  SERVICES+=("app-fpga")
else
  SERVICES+=("app")
fi

echo "==> Starting services from ${COMPOSE_FILE_PATH} ..."
echo "    -> ${SERVICES[*]}"
docker-compose -f "$COMPOSE_FILE_PATH" up -d "${SERVICES[@]}"

echo ""
echo "✅ Services are up."
echo "-----------------------------------"
echo "  - MLflow UI: http://localhost:${PRIVATEER_MLFLOW_SERVER_PORT:-5000}"
if [[ "$ALVEO" == "true" ]]; then
  echo "  - Shell into app-fpga: docker exec -it privateer-ad-app-fpga /bin/bash"
  echo "  - You can also use docker-compose logs -f to find out about connecting to the marimo server"
else
  echo "  - Shell into app:      docker exec -it privateer-ad-app /bin/bash"
fi
echo "-----------------------------------"
echo ""
echo "Logs (follow):   docker-compose logs -f"
echo "Stop services:   docker-compose stop"
echo "Tear down:       docker-compose down"
