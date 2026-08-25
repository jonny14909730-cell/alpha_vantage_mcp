"""Semantic errors for tool dispatch and known upstream failures.

Transport adapters map these classes. They must not infer fault ownership from
TypeError strings or timeout message text.
"""


class ToolError(Exception):
    """Known tool-layer failure with a stable error kind."""

    kind = "tool_error"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


class UnknownToolError(ToolError, ValueError):
    """The named tool is not in the registry (or nested meta-tool target)."""

    kind = "unknown_tool"


class InvalidToolArgumentsError(ToolError):
    """Caller arguments failed schema/signature validation before execution."""

    kind = "invalid_arguments"


class UpstreamTimeoutError(ToolError):
    """A known Alpha Vantage HTTP client timeout. Retryable."""

    kind = "upstream_timeout"
