"""
backend/middleware/logging_middleware.py
─────────────────────────────────────────────────────────────────
FastAPI middleware to log incoming requests, route path, status code,
and response processing time.
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from backend.core.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware logging requests and responses with execution duration.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        path = request.url.path
        method = request.method

        logger.info(f"Incoming request: {method} {path}")

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            logger.info(
                f"Completed request: {method} {path} | "
                f"Status: {response.status_code} | "
                f"Duration: {duration:.4f}s"
            )
            return response
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {method} {path} | "
                f"Error: {str(exc)} | "
                f"Duration: {duration:.4f}s"
            )
            raise exc
