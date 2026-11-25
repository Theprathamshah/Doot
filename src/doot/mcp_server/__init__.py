"""Doot MCP Server package."""

from doot.mcp_server.server import (
    create_mcp_server,
    mcp_get_funnel_stats,
    mcp_get_job_details,
    mcp_list_top_jobs,
    mcp_mark_job_status,
    mcp_tailor_for_job,
)

__all__ = [
    "create_mcp_server",
    "mcp_get_funnel_stats",
    "mcp_get_job_details",
    "mcp_list_top_jobs",
    "mcp_mark_job_status",
    "mcp_tailor_for_job",
]
