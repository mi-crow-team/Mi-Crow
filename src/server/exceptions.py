from __future__ import annotations

"""Custom exception classes for the server application."""


class ServerError(Exception):
    """Base exception for all server errors."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


class ValidationError(ServerError):
    """Raised when request validation fails."""

    pass


class NotFoundError(ServerError):
    """Raised when a requested resource is not found."""

    pass


class ConfigurationError(ServerError):
    """Raised when there's a configuration issue."""

    pass

