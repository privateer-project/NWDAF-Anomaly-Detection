# SQLite Connection and Low-Level Database Operations
#
# This module provides a thin wrapper around sqlite3 for database
# connectivity and basic operations.
#
# Class to implement:
#
# class SQLite:
#   """Low-level SQLite connection manager with DDL bootstrap and WAL mode."""
#   
#   def __init__(self, path: str):
#     """
#     Initialize SQLite manager.
#     Args:
#       - path: Path to SQLite database file
#     
#     Responsibilities:
#       - Store path
#       - Ensure parent directory exists
#       - Check if database file exists
#       - If new database, apply DDL from hitl/ddl.sql
#       - Set PRAGMA journal_mode=WAL
#     """
#   
#   def connect(self) -> sqlite3.Connection:
#     """
#     Create and return a new SQLite connection.
#     
#     Configuration:
#       - Set row_factory = sqlite3.Row for dict-like access
#       - Enable foreign key constraints: PRAGMA foreign_keys=ON
#       - Set timeout for busy database
#     
#     Returns: sqlite3.Connection configured for HITL use
#     """
#   
#   def execute(self, sql: str, params: tuple = ()) -> None:
#     """
#     Execute a single SQL statement (no results expected).
#     
#     Args:
#       - sql: SQL statement (INSERT, UPDATE, DELETE, CREATE, etc.)
#       - params: Tuple of parameters for prepared statement
#     
#     Behavior:
#       - Open connection
#       - Execute statement with params
#       - Commit
#       - Close connection
#       - Wrap sqlite3.Error in DBError with context
#     """
#   
#   def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
#     """
#     Execute query and return single row or None.
#     
#     Args:
#       - sql: SQL SELECT statement
#       - params: Query parameters
#     
#     Returns: Single row as sqlite3.Row or None if no match
#     Raises: DBError on database errors
#     """
#   
#   def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
#     """
#     Execute query and return all matching rows.
#     
#     Args:
#       - sql: SQL SELECT statement
#       - params: Query parameters
#     
#     Returns: List of rows (may be empty)
#     Raises: DBError on database errors
#     """
#   
#   @contextmanager
#   def tx(self) -> Generator[sqlite3.Connection, None, None]:
#     """
#     Context manager for explicit transactions.
#     
#     Usage:
#       with sqlite.tx() as conn:
#         conn.execute("INSERT ...", (...))
#         conn.execute("UPDATE ...", (...))
#       # Auto-commit on success, rollback on exception
#     
#     Behavior:
#       - BEGIN transaction
#       - Yield connection
#       - COMMIT on normal exit
#       - ROLLBACK on exception
#       - Handle SQLite busy errors with retry logic
#     """
#   
#   def _apply_ddl(self) -> None:
#     """
#     Apply DDL schema from hitl/ddl.sql file.
#     
#     Called during __init__ if database is new.
#     Reads SQL file, splits on semicolons, executes each statement.
#     Should be idempotent (uses IF NOT EXISTS).
#     """
#   
#   def _ensure_wal_mode(self) -> None:
#     """
#     Ensure database is in WAL journal mode.
#     Execute: PRAGMA journal_mode=WAL
#     Verify mode was set successfully.
#     """
#
# Helper functions:
#
# - def row_to_dict(row: sqlite3.Row) -> dict:
#   Convert sqlite3.Row to plain dict for serialization
#   Iterate over row.keys() and build dict
#
# Usage patterns:
#   db = SQLite("hitl.db")
#   
#   # Simple execute
#   db.execute("INSERT INTO anomalies ...", (aid, occurred, source, ...))
#   
#   # Query single row
#   row = db.fetchone("SELECT * FROM anomalies WHERE anomaly_id = ?", (aid,))
#   
#   # Query all rows
#   rows = db.fetchall("SELECT * FROM feedback WHERE anomaly_id = ?", (aid,))
#   
#   # Transaction
#   with db.tx() as conn:
#     conn.execute("INSERT INTO anomalies ...", (...))
#     conn.execute("INSERT INTO raw_vectors ...", (...))
