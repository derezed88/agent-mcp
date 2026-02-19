"""
MySQL Database Plugin for MCP Agent

Provides db_query tool for executing SQL queries against MySQL database.
Includes per-table read/write gate management.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from database import execute_sql


class _DbQueryArgs(BaseModel):
    sql: str = Field(description="SQL query to execute")


async def db_query_executor(sql: str) -> str:
    """Execute SQL query."""
    return await execute_sql(sql)


class MysqlPlugin(BasePlugin):
    """MySQL database query plugin."""

    PLUGIN_NAME = "plugin_database_mysql"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "MySQL database query tool with per-table gates"
    DEPENDENCIES = ["mysql-connector-python>=8.0"]
    ENV_VARS = ["MYSQL_USER", "MYSQL_PASS"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize MySQL plugin."""
        try:
            from database import _get_db
            conn = _get_db()
            if not conn.is_connected():
                return False
            self.enabled = True
            return True
        except Exception as e:
            print(f"MySQL plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup MySQL connections."""
        try:
            from database import _db_conn
            if _db_conn and _db_conn.is_connected():
                _db_conn.close()
        except Exception:
            pass
        self.enabled = False

    def get_gate_tools(self) -> Dict[str, Any]:
        """Declare db_query as a gated tool with per-table read/write gates."""
        return {
            "db_query": {
                "type": "db",
                "operations": ["read", "write"],
                "description": "SQL against mymcp (per-table read/write gates)"
            }
        }

    def get_tools(self) -> Dict[str, Any]:
        """Return MySQL tool definitions in LangChain StructuredTool format."""
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=db_query_executor,
                    name="db_query",
                    description="Execute a SQL query against the mymcp MySQL database.",
                    args_schema=_DbQueryArgs,
                )
            ]
        }
