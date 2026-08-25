"""Shared MCP tool-error formatting and Lambda tools/call adapter."""

from __future__ import annotations

import json
from typing import Any

from awslabs.mcp_lambda_handler import MCPLambdaHandler
from awslabs.mcp_lambda_handler.types import ErrorContent
from loguru import logger

from av_api.errors import (
    InvalidToolArgumentsError,
    ToolError,
    UnknownToolError,
    UpstreamTimeoutError,
)
from av_api.registry import validate_tool_arguments


def format_tool_error_text(error: BaseException) -> str:
    """Stable, actionable error text for CallToolResult content."""
    if isinstance(error, ToolError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def mcp_error_result(error: BaseException) -> dict[str, Any]:
    """JSON-serializable CallToolResult for a known execution/client failure."""
    return {
        "content": [{"type": "text", "text": format_tool_error_text(error)}],
        "isError": True,
    }


def log_known_tool_error(tool_name: str, error: BaseException) -> None:
    """Log known client/upstream failures without a traceback."""
    if isinstance(error, InvalidToolArgumentsError):
        logger.warning(f"Invalid arguments for tool {tool_name}: {error}")
        return
    if isinstance(error, UnknownToolError):
        logger.warning(f"Unknown tool {tool_name}: {error}")
        return
    if isinstance(error, UpstreamTimeoutError):
        logger.error(f"Upstream timeout for tool {tool_name}: {error}")
        return
    logger.warning(f"Tool error for {tool_name}: {error}")


class ToolCallMCPLambdaHandler(MCPLambdaHandler):
    """Intercept structurally valid tools/call requests; delegate everything else.

    The locked awslabs 0.1.8 handler does not validate inputSchema and maps every
    tool exception to JSON-RPC -32603 / HTTP 500. This subclass handles only a
    completed tools/call envelope so initialize, tools/list, notifications, and
    malformed requests keep the dependency's existing behavior.
    """

    def handle_request(self, event: dict, context: Any) -> dict:
        intercepted = self._try_handle_tools_call(event)
        if intercepted is not None:
            return intercepted
        return super().handle_request(event, context)

    def _try_handle_tools_call(self, event: dict) -> dict | None:
        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        if headers.get("content-type") != "application/json":
            return None
        if event.get("httpMethod") == "DELETE":
            return None

        try:
            body = json.loads(event.get("body") or "")
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(body, dict):
            return None
        if body.get("jsonrpc") != "2.0" or body.get("method") != "tools/call":
            return None
        if "id" not in body:
            return None

        request_id = body.get("id")
        session_id = headers.get("mcp-session-id")
        try:
            return self._dispatch_recognized_tools_call(body, request_id, session_id)
        except Exception as exc:
            params = body.get("params") if isinstance(body.get("params"), dict) else {}
            name = params.get("name")
            tool_name = name if isinstance(name, str) and name else "unknown"
            logger.exception(f"Unexpected error executing tool {tool_name}")
            error_content = [ErrorContent(text=str(exc)).model_dump()]
            return self._create_error_response(
                -32603,
                f"Error executing tool: {exc}",
                request_id,
                error_content,
                session_id,
            )

    def _dispatch_recognized_tools_call(
        self, body: dict, request_id, session_id: str | None
    ) -> dict | None:
        params = body.get("params")
        if not isinstance(params, dict):
            return None

        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return self._create_error_response(
                -32602,
                "Invalid params: missing tool name",
                request_id,
                session_id=session_id,
            )
        if tool_name not in self.tools:
            return self._create_error_response(
                -32601, f"Tool '{tool_name}' not found", request_id, session_id=session_id
            )

        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return self._create_error_response(
                -32602,
                "Invalid params: arguments must be an object",
                request_id,
                session_id=session_id,
            )
        schema = (self.tools.get(tool_name) or {}).get("inputSchema") or {
            "type": "object"
        }
        try:
            validate_tool_arguments(tool_name, schema, arguments)
        except InvalidToolArgumentsError as exc:
            log_known_tool_error(tool_name, exc)
            return self._create_success_response(
                mcp_error_result(exc), request_id, session_id
            )

        converted_args = self._convert_enum_args(tool_name, arguments)
        try:
            result = self.tool_implementations[tool_name](**converted_args)
        except (InvalidToolArgumentsError, UnknownToolError, UpstreamTimeoutError) as exc:
            log_known_tool_error(tool_name, exc)
            return self._create_success_response(
                mcp_error_result(exc), request_id, session_id
            )

        content = self._convert_result_to_content(result)
        return self._create_success_response(
            {"content": content}, request_id, session_id
        )

    def _convert_enum_args(self, tool_name: str, tool_args: dict) -> dict:
        from enum import Enum
        from typing import get_type_hints

        converted_args = {}
        tool_func = self.tool_implementations[tool_name]
        hints = get_type_hints(tool_func)
        for arg_name, arg_value in tool_args.items():
            arg_type = hints.get(arg_name)
            if isinstance(arg_type, type) and issubclass(arg_type, Enum):
                converted_args[arg_name] = arg_type(arg_value)
            else:
                converted_args[arg_name] = arg_value
        return converted_args
