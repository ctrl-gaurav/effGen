"""
File operations tool for safe file system interactions.

This module provides file operations including read, write, search,
and format conversion with comprehensive security measures.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import json
import csv
import xml.etree.ElementTree as ET
from io import StringIO
import hashlib

from ..base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    ParameterSpec,
    ParameterType,
)


logger = logging.getLogger(__name__)


class FileOperations(BaseTool):
    """
    File operations tool with security restrictions.

    Features:
    - Read files (text, JSON, CSV, XML)
    - Write files (text, JSON, CSV, XML)
    - List directories
    - Search files by pattern
    - Get file metadata
    - Format conversion (JSON <-> CSV, etc.)
    - Safe path handling

    Security:
    - Sandboxed to allowed directories
    - Path traversal prevention
    - File size limits
    - Permission checks
    - Extension whitelisting
    """

    DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    DEFAULT_ENCODING = "utf-8"

    # Allowed file extensions for different operations
    ALLOWED_EXTENSIONS = {
        "read": {".txt", ".json", ".csv", ".xml", ".yaml", ".yml", ".md", ".log"},
        "write": {".txt", ".json", ".csv", ".xml", ".yaml", ".yml", ".md"},
    }

    def __init__(self, allowed_directories: Optional[List[str]] = None):
        """
        Initialize file operations tool.

        Args:
            allowed_directories: List of directories where operations are allowed.
                                If None, uses current working directory.
        """
        super().__init__(
            metadata=ToolMetadata(
                name="file_operations",
                description="Perform safe file system operations including read, write, search, and format conversion",
                category=ToolCategory.FILE_OPERATIONS,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform",
                        required=True,
                        enum=["read", "write", "list", "search", "metadata", "convert"],
                    ),
                    ParameterSpec(
                        name="path",
                        type=ParameterType.STRING,
                        description="File or directory path",
                        required=True,
                    ),
                    ParameterSpec(
                        name="content",
                        type=ParameterType.STRING,
                        description="Content to write (for write operation)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="format",
                        type=ParameterType.STRING,
                        description="File format (text, json, csv, xml)",
                        required=False,
                        default="text",
                        enum=["text", "json", "csv", "xml"],
                    ),
                    ParameterSpec(
                        name="encoding",
                        type=ParameterType.STRING,
                        description="Text encoding",
                        required=False,
                        default="utf-8",
                    ),
                    ParameterSpec(
                        name="pattern",
                        type=ParameterType.STRING,
                        description="Search pattern (for search operation)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="recursive",
                        type=ParameterType.BOOLEAN,
                        description="Whether to search recursively",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="target_format",
                        type=ParameterType.STRING,
                        description="Target format for conversion",
                        required=False,
                        enum=["json", "csv", "xml"],
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {"type": "any"},
                        "message": {"type": "string"},
                    },
                },
                timeout_seconds=30,
                tags=["file", "filesystem", "io", "data"],
                examples=[
                    {
                        "operation": "read",
                        "path": "/path/to/file.txt",
                        "format": "text",
                    },
                    {
                        "operation": "write",
                        "path": "/path/to/output.json",
                        "content": '{"key": "value"}',
                        "format": "json",
                    },
                ],
            )
        )
        self._allowed_directories = self._normalize_allowed_directories(allowed_directories)
        self._max_file_size = self.DEFAULT_MAX_FILE_SIZE

    def _normalize_allowed_directories(
        self, directories: Optional[List[str]]
    ) -> List[Path]:
        """Normalize and validate allowed directories."""
        if not directories:
            return [Path.cwd()]

        normalized = []
        for dir_path in directories:
            path = Path(dir_path).resolve()
            if path.exists() and path.is_dir():
                normalized.append(path)
            else:
                logger.warning(f"Directory does not exist or is not a directory: {dir_path}")

        return normalized or [Path.cwd()]

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is within allowed directories."""
        try:
            resolved_path = path.resolve()
            return any(
                resolved_path == allowed_dir or allowed_dir in resolved_path.parents
                for allowed_dir in self._allowed_directories
            )
        except (OSError, RuntimeError):
            return False

    def _validate_path(self, path: str, operation: str) -> Path:
        """
        Validate and resolve path.

        Args:
            path: Path to validate
            operation: Operation being performed

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or not allowed
        """
        # Convert to Path and resolve
        try:
            file_path = Path(path).resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")

        # Check if path is allowed
        if not self._is_path_allowed(file_path):
            raise ValueError(
                f"Path '{path}' is outside allowed directories: {self._allowed_directories}"
            )

        # Check extension for read/write operations
        if operation in ("read", "write"):
            allowed_exts = self.ALLOWED_EXTENSIONS.get(operation, set())
            if file_path.suffix.lower() not in allowed_exts and allowed_exts:
                raise ValueError(
                    f"File extension '{file_path.suffix}' not allowed for {operation} operation"
                )

        return file_path

    async def _execute(
        self,
        operation: str,
        path: str,
        content: Optional[str] = None,
        format: str = "text",
        encoding: str = DEFAULT_ENCODING,
        pattern: Optional[str] = None,
        recursive: bool = False,
        target_format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute file operation.

        Args:
            operation: Operation to perform
            path: File or directory path
            content: Content to write
            format: File format
            encoding: Text encoding
            pattern: Search pattern
            recursive: Recursive search flag
            target_format: Target format for conversion

        Returns:
            Dict with success status, data, and message
        """
        operations = {
            "read": self._read_file,
            "write": self._write_file,
            "list": self._list_directory,
            "search": self._search_files,
            "metadata": self._get_metadata,
            "convert": self._convert_format,
        }

        handler = operations.get(operation)
        if not handler:
            return {
                "success": False,
                "data": None,
                "message": f"Unknown operation: {operation}",
            }

        try:
            if operation == "read":
                data = await handler(path, format, encoding)
            elif operation == "write":
                data = await handler(path, content, format, encoding)
            elif operation == "list":
                data = await handler(path, recursive)
            elif operation == "search":
                data = await handler(path, pattern, recursive)
            elif operation == "metadata":
                data = await handler(path)
            elif operation == "convert":
                data = await handler(path, format, target_format, encoding)

            return {
                "success": True,
                "data": data,
                "message": f"Operation '{operation}' completed successfully",
            }

        except Exception as e:
            # File operation errors are often expected (path restrictions, permissions, etc.)
            logger.debug(f"File operation failed: {e}")
            return {
                "success": False,
                "data": None,
                "message": f"Operation failed: {str(e)}",
            }

    async def _read_file(
        self, path: str, format: str, encoding: str
    ) -> Union[str, Dict, List]:
        """Read file with format parsing."""
        file_path = self._validate_path(path, "read")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self._max_file_size:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds limit ({self._max_file_size} bytes)"
            )

        # Read file content
        def read_sync():
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()

        content = await asyncio.to_thread(read_sync)

        # Parse based on format
        if format == "json":
            return json.loads(content)
        elif format == "csv":
            reader = csv.DictReader(StringIO(content))
            return list(reader)
        elif format == "xml":
            root = ET.fromstring(content)
            return self._xml_to_dict(root)
        else:
            return content

    async def _write_file(
        self, path: str, content: Optional[str], format: str, encoding: str
    ) -> str:
        """Write content to file with format conversion."""
        if content is None:
            raise ValueError("Content is required for write operation")

        file_path = self._validate_path(path, "write")

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Format content
        if format == "json":
            # Validate JSON
            data = json.loads(content) if isinstance(content, str) else content
            formatted_content = json.dumps(data, indent=2, ensure_ascii=False)
        elif format == "csv":
            # Content should be list of dicts
            data = json.loads(content) if isinstance(content, str) else content
            if not isinstance(data, list):
                raise ValueError("CSV format requires list of dictionaries")
            output = StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            formatted_content = output.getvalue()
        else:
            formatted_content = content

        # Write file
        def write_sync():
            with open(file_path, "w", encoding=encoding) as f:
                f.write(formatted_content)

        await asyncio.to_thread(write_sync)

        return str(file_path)

    async def _list_directory(self, path: str, recursive: bool) -> List[Dict[str, Any]]:
        """List directory contents."""
        dir_path = self._validate_path(path, "list")

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        def list_sync():
            entries = []
            if recursive:
                for item in dir_path.rglob("*"):
                    if self._is_path_allowed(item):
                        entries.append(self._get_file_info(item))
            else:
                for item in dir_path.iterdir():
                    if self._is_path_allowed(item):
                        entries.append(self._get_file_info(item))
            return entries

        return await asyncio.to_thread(list_sync)

    async def _search_files(
        self, path: str, pattern: Optional[str], recursive: bool
    ) -> List[str]:
        """Search for files matching pattern."""
        dir_path = self._validate_path(path, "search")

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not pattern:
            raise ValueError("Pattern is required for search operation")

        def search_sync():
            matches = []
            search_method = dir_path.rglob if recursive else dir_path.glob
            for item in search_method(pattern):
                if self._is_path_allowed(item):
                    matches.append(str(item))
            return matches

        return await asyncio.to_thread(search_sync)

    async def _get_metadata(self, path: str) -> Dict[str, Any]:
        """Get file metadata."""
        file_path = self._validate_path(path, "metadata")

        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        return self._get_file_info(file_path)

    async def _convert_format(
        self, path: str, source_format: str, target_format: Optional[str], encoding: str
    ) -> str:
        """Convert file from one format to another."""
        if not target_format:
            raise ValueError("Target format is required for convert operation")

        # Read in source format
        content = await self._read_file(path, source_format, encoding)

        # Generate new filename
        file_path = Path(path)
        new_path = file_path.with_suffix(f".{target_format}")

        # Write in target format
        await self._write_file(
            str(new_path),
            json.dumps(content) if target_format != "text" else str(content),
            target_format,
            encoding,
        )

        return str(new_path)

    def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get file information."""
        stat = path.stat()
        return {
            "path": str(path),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "extension": path.suffix,
        }

    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        if element.attrib:
            result["@attributes"] = element.attrib

        if element.text and element.text.strip():
            result["text"] = element.text.strip()

        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return {element.tag: result} if result else {element.tag: element.text}

    def set_max_file_size(self, size_bytes: int) -> None:
        """Set maximum allowed file size for read operations."""
        self._max_file_size = size_bytes

    def add_allowed_directory(self, directory: str) -> None:
        """Add a directory to the allowed list."""
        path = Path(directory).resolve()
        if path.exists() and path.is_dir():
            self._allowed_directories.append(path)
        else:
            raise ValueError(f"Invalid directory: {directory}")
