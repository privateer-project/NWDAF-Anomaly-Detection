# Command-Line Interface
#
# This module provides the CLI for HITL system using argparse or Typer.
# All HITL operations are accessible via command-line.
#
# Implementation approach (choose one):
# 1. argparse - standard library, more verbose
# 2. Typer - modern, type-hint based, cleaner syntax
#
# Recommended: Typer for better DX
#
# # Setup
# import typer
# from typing_extensions import Annotated
# 
# app = typer.Typer(
#   name="hitl",
#   help="HITL - Human-in-the-Loop Anomaly Filtering CLI",
#   no_args_is_help=True
# )
#
# # Global state/config
# # Load config once and pass to commands
# def get_hitl() -> HITL:
#   """Helper to create HITL instance."""
#   config = get_env_config()
#   return HITL(config)
#
# # ===== COMMANDS =====
#
# @app.command()
# def init_db():
#   """
#   Initialize database with schema.
#   
#   Creates database file and applies DDL.
#   Safe to run multiple times (idempotent).
#   
#   Example:
#     hitl init-db
#   """
#
# @app.command()
# def upsert(
#   anomaly_id: Annotated[str, typer.Option("--id", help="Anomaly ID")],
#   source: Annotated[str, typer.Option("--source", help="Source/unit")],
#   occurred_at: Annotated[str, typer.Option("--when", help="ISO timestamp")],
#   npy_file: Annotated[str, typer.Option("--npy", help="Path to .npy file")],
#   dtype: Annotated[str, typer.Option("--dtype", help="NumPy dtype")] = "float32"
# ):
#   """
#   Insert or update anomaly with feature vector.
#   
#   Loads tensor from .npy file and stores in database.
#   
#   Example:
#     hitl upsert --id A1 --source unit-1 --when 2025-11-04T10:00:00Z --npy data.npy
#   """
#
# @app.command()
# def feedback(
#   anomaly_id: Annotated[str, typer.Option("--anomaly", help="Anomaly ID")],
#   user_id: Annotated[str, typer.Option("--user", help="User ID")],
#   label: Annotated[str, typer.Option("--label", help="Label: TP, FP, etc.")],
#   confidence: Annotated[float | None, typer.Option("--confidence", help="Confidence 0-1")] = None,
#   note: Annotated[str | None, typer.Option("--note", help="Optional note")] = None
# ):
#   """
#   Submit human feedback on anomaly.
#   
#   Example:
#     hitl feedback --anomaly A1 --user analyst-1 --label TP --confidence 0.9
#   """
#
# @app.command()
# def train(
#   mode: Annotated[str | None, typer.Option("--mode", help="dense or conv1d")] = None,
#   schema_id: Annotated[str | None, typer.Option("--schema", help="Schema ID")] = None,
#   epochs: Annotated[int, typer.Option("--epochs", help="Training epochs")] = 100,
#   lr: Annotated[float, typer.Option("--lr", help="Learning rate")] = 0.001,
#   batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 32
# ):
#   """
#   Train new anomaly detection model.
#   
#   If schema_id not provided, uses all available data (must be single schema).
#   
#   Example:
#     hitl train --mode dense --epochs 100 --lr 0.001
#   """
#
# @app.command()
# def set_live(
#   model_version: Annotated[str, typer.Option("--model", help="Model version")]
# ):
#   """
#   Set which model is used for inference.
#   
#   Example:
#     hitl set-live --model AE-2025.11.04-1
#   """
#
# @app.command()
# def predict(
#   npy_file: Annotated[str | None, typer.Option("--npy", help="Path to .npy file")] = None,
#   anomaly_id: Annotated[str | None, typer.Option("--anomaly", help="Anomaly ID")] = None
# ):
#   """
#   Predict whether anomaly is true positive.
#   
#   Provide either --npy or --anomaly (not both).
#   
#   Example:
#     hitl predict --npy data.npy
#     hitl predict --anomaly A1
#   """
#
# @app.command()
# def list_anomalies(
#   limit: Annotated[int, typer.Option("--limit", help="Max results")] = 100,
#   source: Annotated[str | None, typer.Option("--source", help="Filter by source")] = None
# ):
#   """
#   List anomalies from database.
#   
#   Example:
#     hitl list-anomalies --limit 50 --source unit-1
#   """
#
# @app.command()
# def list_models(
#   kind: Annotated[str | None, typer.Option("--kind", help="Filter by mode")] = None
# ):
#   """
#   List trained models.
#   
#   Example:
#     hitl list-models --kind dense
#   """
#
# @app.command()
# def stats():
#   """
#   Show system statistics.
#   
#   Displays counts of anomalies, models, feedback, etc.
#   
#   Example:
#     hitl stats
#   """
#
# @app.command()
# def health():
#   """
#   Check system health.
#   
#   Verifies database, artifacts, and live model status.
#   
#   Example:
#     hitl health
#   """
#
# @app.command()
# def get_anomaly(
#   anomaly_id: Annotated[str, typer.Argument(help="Anomaly ID")]
# ):
#   """
#   Get details for specific anomaly.
#   
#   Example:
#     hitl get-anomaly A1
#   """
#
# @app.command()
# def export_vector(
#   anomaly_id: Annotated[str, typer.Option("--anomaly", help="Anomaly ID")],
#   output: Annotated[str, typer.Option("--output", "-o", help="Output .npy file")]
# ):
#   """
#   Export feature vector to .npy file.
#   
#   Example:
#     hitl export-vector --anomaly A1 --output vector.npy
#   """
#
# # ===== HELPER FUNCTIONS =====
#
# def load_npy(path: str) -> np.ndarray:
#   """Load NumPy array from .npy file."""
#   # Use np.load() with allow_pickle=False
#
# def save_npy(arr: np.ndarray, path: str) -> None:
#   """Save NumPy array to .npy file."""
#   # Use np.save()
#
# def print_json(data: dict) -> None:
#   """Print data as formatted JSON."""
#   # Use json.dumps() with indent=2
#   # Or use rich library for colored output
#
# def print_table(data: list[dict], columns: list[str]) -> None:
#   """Print data as table."""
#   # Use tabulate library or rich.table
#   # Or simple formatting
#
# def handle_error(exc: Exception) -> None:
#   """
#   Handle and display errors appropriately.
#   
#   Different error types get different messages/exit codes.
#   """
#
# # ===== MAIN ENTRY POINT =====
#
# def main():
#   """Main entry point for CLI."""
#   try:
#     app()
#   except KeyboardInterrupt:
#     typer.echo("\nCancelled by user")
#     raise typer.Exit(130)
#   except Exception as e:
#     handle_error(e)
#     raise typer.Exit(1)
#
# if __name__ == "__main__":
#   main()
#
# # Usage examples:
# #
# # Initialize database:
# #   hitl init-db
# #
# # Add anomaly:
# #   hitl upsert --id A1 --source unit-1 --when 2025-11-04T10:00:00Z --npy data.npy
# #
# # Submit feedback:
# #   hitl feedback --anomaly A1 --user analyst-1 --label TP --confidence 0.9
# #
# # Train model:
# #   hitl train --mode dense --epochs 100
# #
# # Set live model:
# #   hitl set-live --model AE-2025.11.04-1
# #
# # Predict:
# #   hitl predict --anomaly A1
# #   hitl predict --npy new_data.npy
# #
# # List operations:
# #   hitl list-anomalies --limit 10
# #   hitl list-models
# #
# # System info:
# #   hitl stats
# #   hitl health
# #
# # Get specific anomaly:
# #   hitl get-anomaly A1
# #
# # Export vector:
# #   hitl export-vector --anomaly A1 --output vector.npy
