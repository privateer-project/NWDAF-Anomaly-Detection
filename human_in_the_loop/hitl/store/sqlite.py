"""
SQLite connection manager for HITL system.

This module provides a thin wrapper around sqlite3 for database connectivity
and basic operations, including DDL bootstrap and WAL mode configuration.

Classes:
    SQLite: Low-level SQLite connection manager

Functions:
    row_to_dict: Convert sqlite3.Row to dict
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from ..errors import DBError


class SQLite:
    """
    SQLite connection manager with DDL bootstrap and WAL mode.
    
    Handles database initialization, connection management, and low-level
    operations. Automatically applies DDL schema on first run.
    
    Example:
        >>> db = SQLite("hitl.db")
        >>> db.execute("INSERT INTO settings VALUES (?, ?)", ("key", "value"))
        >>> row = db.fetchone("SELECT * FROM settings WHERE key = ?", ("key",))
    """
    
    def __init__(self, path: str | Path):
        """
        Initialize SQLite manager.
        
        Args:
            path: Path to SQLite database file
        
        Side effects:
            - Creates parent directories if needed
            - Applies DDL schema if database is new
            - Enables WAL mode
        """
        self.path = Path(path).resolve()
        is_new = not self.path.exists()
        
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply DDL if new database
        if is_new:
            self._apply_ddl()
        
        # Ensure WAL mode is enabled
        self._ensure_wal_mode()
    
    def connect(self) -> sqlite3.Connection:
        """
        Create and return a new SQLite connection.
        
        Returns:
            sqlite3.Connection: Configured connection with Row factory
        
        Configuration:
            - row_factory = sqlite3.Row for dict-like access
            - PRAGMA foreign_keys=ON
            - timeout = 10.0 seconds
        
        Example:
            >>> conn = db.connect()
            >>> row = conn.execute("SELECT * FROM settings").fetchone()
            >>> conn.close()
        """
        try:
            conn = sqlite3.connect(str(self.path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except sqlite3.Error as e:
            raise DBError(f"Failed to connect to database: {e}") from e
    
    def execute(self, sql: str, params: tuple = ()) -> None:
        """
        Execute a single SQL statement (no results expected).
        
        Args:
            sql: SQL statement (INSERT, UPDATE, DELETE, etc.)
            params: Parameters for prepared statement
        
        Raises:
            DBError: On database errors
        
        Example:
            >>> db.execute("INSERT INTO settings VALUES (?, ?)", ("key", "value"))
        """
        try:
            with self.connect() as conn:
                conn.execute(sql, params)
                conn.commit()
        except sqlite3.Error as e:
            raise DBError(f"Failed to execute statement: {e}") from e
    
    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """
        Execute query and return single row or None.
        
        Args:
            sql: SQL SELECT statement
            params: Query parameters
        
        Returns:
            Single row as sqlite3.Row or None if no match
        
        Raises:
            DBError: On database errors
        
        Example:
            >>> row = db.fetchone("SELECT * FROM settings WHERE key = ?", ("key",))
            >>> if row:
            ...     print(row["value"])
        """
        try:
            with self.connect() as conn:
                cursor = conn.execute(sql, params)
                return cursor.fetchone()
        except sqlite3.Error as e:
            raise DBError(f"Failed to fetch row: {e}") from e
    
    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """
        Execute query and return all matching rows.
        
        Args:
            sql: SQL SELECT statement
            params: Query parameters
        
        Returns:
            List of rows (may be empty)
        
        Raises:
            DBError: On database errors
        
        Example:
            >>> rows = db.fetchall("SELECT * FROM settings")
            >>> for row in rows:
            ...     print(row["key"], row["value"])
        """
        try:
            with self.connect() as conn:
                cursor = conn.execute(sql, params)
                return cursor.fetchall()
        except sqlite3.Error as e:
            raise DBError(f"Failed to fetch rows: {e}") from e
    
    @contextmanager
    def tx(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for explicit transactions.
        
        Yields:
            sqlite3.Connection: Connection in transaction mode
        
        Behavior:
            - Begins transaction
            - Yields connection
            - Commits on normal exit
            - Rolls back on exception
        
        Example:
            >>> with db.tx() as conn:
            ...     conn.execute("INSERT INTO settings VALUES (?, ?)", ("k1", "v1"))
            ...     conn.execute("INSERT INTO settings VALUES (?, ?)", ("k2", "v2"))
            ... # Both inserts committed or both rolled back
        """
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise DBError(f"Transaction failed: {e}") from e
        finally:
            conn.close()
    
    def _apply_ddl(self) -> None:
        """
        Apply DDL schema from hitl/ddl.sql file.
        
        Reads SQL file, executes all statements. Should be idempotent
        (uses IF NOT EXISTS).
        
        Raises:
            DBError: If DDL file not found or SQL execution fails
        """
        # Find ddl.sql relative to this file
        ddl_path = Path(__file__).parent.parent / "ddl.sql"
        
        if not ddl_path.exists():
            raise DBError(f"DDL file not found: {ddl_path}")
        
        try:
            ddl_sql = ddl_path.read_text()
            
            with self.connect() as conn:
                # Execute all statements in the DDL file
                conn.executescript(ddl_sql)
                conn.commit()
        except sqlite3.Error as e:
            raise DBError(f"Failed to apply DDL: {e}") from e
    
    def _ensure_wal_mode(self) -> None:
        """
        Ensure database is in WAL journal mode.
        
        WAL mode allows concurrent readers while a writer is active.
        
        Raises:
            DBError: If WAL mode cannot be enabled
        """
        try:
            with self.connect() as conn:
                cursor = conn.execute("PRAGMA journal_mode=WAL")
                mode = cursor.fetchone()[0]
                if mode.upper() != "WAL":
                    raise DBError(f"Failed to enable WAL mode, got: {mode}")
        except sqlite3.Error as e:
            raise DBError(f"Failed to set WAL mode: {e}") from e


def row_to_dict(row: sqlite3.Row) -> dict:
    """
    Convert sqlite3.Row to plain dict.
    
    Args:
        row: sqlite3.Row object
    
    Returns:
        dict: Plain dictionary with same keys/values
    
    Example:
        >>> row = db.fetchone("SELECT * FROM settings WHERE key = 'test'")
        >>> data = row_to_dict(row)
        >>> print(data)
        {'key': 'test', 'value': 'example'}
    """
    return {key: row[key] for key in row.keys()}
