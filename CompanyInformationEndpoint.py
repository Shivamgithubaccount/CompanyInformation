"""
Company Information API
=======================

Single-file FastAPI service that retrieves company profile information from
Yahoo Finance through yfinance.

Install:
    pip install fastapi uvicorn yfinance

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Example requests:
    GET http://127.0.0.1:8000/company/AAPL
    GET http://127.0.0.1:8000/company/RELIANCE.NS
    GET http://127.0.0.1:8000/health

Interactive API docs:
    http://127.0.0.1:8000/docs

Notes:
- Yahoo Finance/yfinance availability, fields, and rate limits are not guaranteed.
- This is a single-process implementation. For production deployments with
  multiple workers/instances, use a shared rate limiter and shared cache.
- Do not expose this service publicly without appropriate authentication,
  request limits, monitoring, and deployment-level timeout controls.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "Company Information API"
APP_VERSION = "1.0.0"

# A small per-process limit to reduce accidental bursts. This is NOT a
# replacement for a distributed rate limiter in a multi-worker deployment.
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Accepted examples: AAPL, BRK-B, RELIANCE.NS, ^GSPC
# This validates the shape of a symbol, not whether Yahoo Finance lists it.
SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9^._=-]{1,32}$")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("company_information_api")


# ---------------------------------------------------------------------------
# API and response schemas
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Retrieve company profile information from Yahoo Finance using a "
        "ticker symbol. Fields may be unavailable for some symbols."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

class Officer(BaseModel):
    """A company officer record supplied by Yahoo Finance."""

    name: str
    title: str | None = None

class CompanyInformation(BaseModel):
    """Stable, documented response contract for a company profile."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    company_name: str
    business_summary: str | None = None
    industry: str | None = None
    sector: str | None = None
    key_officers: list[Officer]
    website: str | None = None
    country: str | None = None
    exchange: str | None = None
    currency: str | None = None
    source: str = "Yahoo Finance via yfinance"
    retrieved_at: datetime

class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Lightweight in-memory rate limiter
# ---------------------------------------------------------------------------

_rate_limit_state: dict[str, list[float]] = {}

@app.middleware("http")
async def simple_rate_limit(request: Request, call_next):
    """
    Apply a basic per-client-IP request limit.

    In-memory state is per process and resets on restart. For multi-worker or
    multi-instance production deployments, replace this with Redis/API-gateway
    rate limiting. Health checks and docs are limited too.
    """
    if RATE_LIMIT_REQUESTS <= 0:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    timestamps = _rate_limit_state.get(client_ip, [])
    timestamps = [stamp for stamp in timestamps if stamp > window_start]

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        _rate_limit_state[client_ip] = timestamps
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please retry later."},
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )

    timestamps.append(now)
    _rate_limit_state[client_ip] = timestamps

    # Basic cleanup to avoid unbounded stale entries.
    if len(_rate_limit_state) > 10_000:
        for ip in list(_rate_limit_state):
            kept = [stamp for stamp in _rate_limit_state[ip]
                    if stamp > window_start]
            if kept:
                _rate_limit_state[ip] = kept
            else:
                _rate_limit_state.pop(ip, None)

    return await call_next(request)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Return consistent JSON for request/path validation failures."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request parameters.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log unexpected errors without exposing internals to API consumers."""
    logger.exception("Unhandled error for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred."},
    )


# ---------------------------------------------------------------------------
# Yahoo Finance data access and normalization
# ---------------------------------------------------------------------------

def _clean_text(value: Any) -> str | None:
    """Normalize optional values into non-empty strings."""
    if value is None:
        return None
    try:
        text_value = str(value).strip()
    except Exception:
        return None
    return text_value or None


def _normalize_officers(raw_officers: Any) -> list[Officer]:
    """
    Convert Yahoo Finance's companyOfficers field into name/title records.

    Yahoo Finance can omit this field, return an empty list, or change its
    contents. Only the requested public name/title fields are exposed.
    """
    if not isinstance(raw_officers, (list, tuple)):
        return []

    officers: list[Officer] = []
    seen: set[tuple[str, str | None]] = set()

    for item in raw_officers:
        if not isinstance(item, dict):
            continue

        name = _clean_text(item.get("name"))
        title = _clean_text(item.get("title"))

        # Do not emit nameless entries; they are not useful as officer records.
        if not name:
            continue

        identity = (name.casefold(), title.casefold() if title else None)
        if identity in seen:
            continue

        seen.add(identity)
        officers.append(Officer(name=name, title=title))

    return officers


def _fetch_company_information(symbol: str) -> dict[str, Any]:
    """
    Synchronous Yahoo Finance call, intended to run in FastAPI's thread pool.

    yfinance's .info property may perform network requests and can fail because
    of upstream outages, network issues, or Yahoo Finance rate limiting.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
    except Exception as exc:
        message = str(exc).lower()
        logger.warning("Yahoo Finance lookup failed for %s: %s", symbol, exc)

        # Avoid exposing raw upstream exception details to the API caller.
        if "rate limit" in message or "too many requests" in message:
            raise HTTPException(
                status_code=503,
                detail="Yahoo Finance is rate limiting requests. Please retry later.",
                headers={"Retry-After": "30"},
            ) from exc

        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve company information from Yahoo Finance.",
        ) from exc

    if not isinstance(info, dict) or not info:
        raise HTTPException(
            status_code=404,
            detail=f"No company information was found for symbol '{symbol}'.",
        )

    company_name = (
        _clean_text(info.get("longName"))
        or _clean_text(info.get("shortName"))
        or _clean_text(info.get("displayName"))
    )

    # A non-empty dict containing no company identity may be an incomplete
    # upstream response rather than a valid company profile.
    if not company_name:
        raise HTTPException(
            status_code=404,
            detail=f"No usable company profile was found for symbol '{symbol}'.",
        )

    return {
        "symbol": symbol,
        "company_name": company_name,
        "business_summary": _clean_text(info.get("longBusinessSummary")),
        "industry": _clean_text(info.get("industry")),
        "sector": _clean_text(info.get("sector")),
        "key_officers": _normalize_officers(info.get("companyOfficers")),
        "website": _clean_text(info.get("website")),
        "country": _clean_text(info.get("country")),
        "exchange": (
            _clean_text(info.get("fullExchangeName"))
            or _clean_text(info.get("exchange"))
        ),
        "currency": _clean_text(info.get("currency")),
        "source": "Yahoo Finance via yfinance",
        "retrieved_at": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/company/{symbol}",
    response_model=CompanyInformation,
    responses={
        404: {"model": ErrorResponse, "description": "No company profile found."},
        422: {"description": "Malformed ticker symbol."},
        429: {"model": ErrorResponse, "description": "API request rate limit exceeded."},
        502: {"model": ErrorResponse, "description": "Yahoo Finance upstream failure."},
        503: {"model": ErrorResponse, "description": "Yahoo Finance is temporarily unavailable."},
    },
    summary="Retrieve detailed company information",
)
async def get_company_information(
    symbol: str = Path(
        ...,
        min_length=1,
        max_length=32,
        description="Yahoo Finance ticker symbol, e.g. AAPL or RELIANCE.NS",
        examples=["AAPL"],
    ),
) -> CompanyInformation:
    """
    Retrieve company name, business summary, sector, industry, and key officers.

    HTTP status codes:
      - 200: Profile retrieved.
      - 404: No usable company profile for the symbol.
      - 422: Invalid symbol format.
      - 429: This API's per-process request limit was exceeded.
      - 502: Yahoo Finance request failed.
      - 503: Yahoo Finance rate limiting was detected.
    """
    normalized_symbol = symbol.strip().upper()

    if not SYMBOL_PATTERN.fullmatch(normalized_symbol):
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid ticker symbol format. Use 1-32 letters, digits, "
                "or supported ticker characters (., ^, _, =, -)."
            ),
        )

    # Run blocking yfinance work outside the async event loop.
    result = await run_in_threadpool(
        _fetch_company_information,
        normalized_symbol,
    )
    return CompanyInformation(**result)


@app.get("/health", summary="Health check")
async def health_check() -> dict[str, str]:
    """
    Liveness check only. It does not make a Yahoo Finance network request and
    therefore does not guarantee that the upstream data provider is available.
    """
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Small landing response pointing users to the API documentation."""
    return {
        "service": APP_NAME,
        "docs": "/docs",
        "company_endpoint": "/company/{symbol}",
        "health_endpoint": "/health",
    }


if __name__ == "__main__":
    # Allows `python main.py` as well as `uvicorn main:app`.
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
