"""
Google Drive Plugin for MCP Agent

Provides google_drive tool for CRUD operations on Google Drive within authorized folder.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from drive import run_drive_op


class _GoogleDriveArgs(BaseModel):
    operation: Literal["list", "create", "read", "append", "delete"] = Field(
        description="Operation to perform"
    )
    file_id: Optional[str] = Field(default="", description="File ID for read/append/delete operations")
    file_name: Optional[str] = Field(default="", description="File name for create operation")
    content: Optional[str] = Field(default="", description="Content for create/append operations")
    folder_id: Optional[str] = Field(
        default="",
        description="Leave empty to use configured folder. Do NOT use 'root'."
    )


async def google_drive_executor(
    operation: str,
    file_id: str = "",
    file_name: str = "",
    content: str = "",
    folder_id: str = ""
) -> str:
    """Execute Google Drive operation."""
    return await run_drive_op(
        operation,
        file_id or None,
        file_name or None,
        content or None,
        folder_id or None
    )


class GoogleDrivePlugin(BasePlugin):
    """Google Drive CRUD operations plugin."""

    PLUGIN_NAME = "plugin_storage_googledrive"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Google Drive CRUD operations within authorized folder"
    DEPENDENCIES = ["google-auth", "google-auth-oauthlib", "google-api-python-client"]
    ENV_VARS = ["FOLDER_ID"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        """Initialize Google Drive plugin."""
        try:
            import os
            from config import DRIVE_FOLDER_ID, DRIVE_CREDS_FILE

            if not DRIVE_FOLDER_ID:
                print("Google Drive plugin: FOLDER_ID not set in .env")
                return False

            if not os.path.exists(DRIVE_CREDS_FILE):
                print(f"Google Drive plugin: credentials.json not found")
                return False

            from drive import _get_drive_service
            self.enabled = True
            return True
        except Exception as e:
            print(f"Google Drive plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        """Cleanup Google Drive resources."""
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        """Return Google Drive tool definitions in LangChain StructuredTool format."""
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=google_drive_executor,
                    name="google_drive",
                    description=(
                        "CRUD operations on Google Drive within a SPECIFIC authorized folder. "
                        "Second-level fallback. "
                        "IMPORTANT: Only accesses files in the pre-configured FOLDER_ID. "
                        "Do NOT pass folder_id='root'. Leave folder_id empty to use the configured folder."
                    ),
                    args_schema=_GoogleDriveArgs,
                )
            ]
        }
