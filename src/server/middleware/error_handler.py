from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from fastapi import HTTPException, status

from server.exceptions import NotFoundError, ServerError, ValidationError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(func: F) -> F:
    """
    Decorator to handle errors consistently across route handlers.

    Converts server exceptions to appropriate HTTP exceptions.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ValidationError as exc:
            logger.debug("Validation error", exc_info=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except NotFoundError as exc:
            logger.debug("Not found error", exc_info=True)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            # ValueError is commonly used for validation, treat as 400
            logger.debug("Value error", exc_info=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except ServerError as exc:
            logger.warning("Server error", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Unexpected error", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(exc)}",
            ) from exc

    return wrapper  # type: ignore[return-value]

