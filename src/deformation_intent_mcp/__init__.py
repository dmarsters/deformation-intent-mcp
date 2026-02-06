"""DeformationIntent MCP Server - Classify deformation types and recommend transformations."""

from .server import mcp, create_server

__version__ = "1.0.0"
__all__ = ["mcp", "create_server"]
