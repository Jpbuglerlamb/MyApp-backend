# jobs/adzuna.py
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Tuple

import httpx

from settings import ADZUNA_APP_ID, ADZUNA_APP_KEY
from ai.extraction import normalize_role_for_api

_CACHE: Dict[Tuple[str, str, str], Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_TTL_SECONDS = 60.0


def _cache_key(query: str, location: str, income_type: str) -> Tuple[str, str, str]:
    return (query.strip().lower(), location.strip().lower(), income_type.strip().lower())


def _get_cached(query: str, location: str, income_type: str) -> List[Dict[str, Any]] | None:
    key = _cache_key(query, location, income_type)
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, data = entry
    if time.time() - ts <= _CACHE_TTL_SECONDS:
        return data
    _CACHE.pop(key, None)
    return None


def _set_cached(query: str, location: str, income_type: str, data: List[Dict[str, Any]]) -> None:
    key = _cache_key(query, location, income_type)
    _CACHE[key] = (time.time(), data)


def _safe_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""


async def _get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    params: Dict[str, Any],
    tries: int = 5,
    base_delay: float = 1.0,
) -> Dict[str, Any]:
    delay = base_delay
    for attempt in range(tries):
        resp = await client.get(url, params=params)

        if resp.status_code < 400:
            return resp.json()

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_s = float(retry_after)
                except ValueError:
                    wait_s = delay + random.uniform(0, 0.5)
            else:
                wait_s = delay + random.uniform(0, 0.5)

            print(f"[DEBUG] Adzuna 429. Sleeping {wait_s:.1f}s (attempt {attempt+1}/{tries})")
            await asyncio.sleep(wait_s)
            delay *= 2
            continue

        resp.raise_for_status()

    raise RuntimeError("Adzuna rate-limited (429) after retries")


def _build_what(role_keywords: str) -> str:
    # Keep 'what' purely about the role keywords.
    return normalize_role_for_api(role_keywords).strip()


def _income_params(income_type: str) -> Dict[str, Any]:
    """
    Use Adzuna's official filters (NOT keywords).
    Docs show full_time=1 usage.  [oai_citation:1‡Adzuna API](https://developer.adzuna.com/docs/search)
    """
    t = (income_type or "").strip().lower()
    if t == "full-time":
        return {"full_time": 1}
    if t == "part-time":
        return {"part_time": 1}
    return {}


async def fetch_jobs(role_keywords: str, location: str, income_type: str = "job") -> List[Dict[str, Any]]:
    role_keywords = (role_keywords or "").strip()
    location = (location or "").strip()
    if not role_keywords or not location:
        return []

    what = _build_what(role_keywords)

    cached = _get_cached(what, location, income_type)
    if cached is not None:
        print(f"[DEBUG] Adzuna cache hit: what='{what}' where='{location}' income='{income_type}'")
        return cached

    url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"

    params: Dict[str, Any] = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": what,
        "where": location,
        "results_per_page": 30,
        "content-type": "application/json",
        **_income_params(income_type),
    }

    print(f"[DEBUG] Adzuna request: what='{what}' where='{location}' income='{income_type}' params_income={_income_params(income_type)}")

    try:
        async with httpx.AsyncClient(timeout=15) as client_http:
            data = await _get_with_backoff(client_http, url, params, tries=5)
            results = (data.get("results") or [])

            normalized_jobs: List[Dict[str, Any]] = []
            for job in results:
                title = _safe_str(job.get("title"))

                company_val = job.get("company")
                company = _safe_str(company_val.get("display_name")) if isinstance(company_val, dict) else _safe_str(company_val)

                location_val = job.get("location")
                loc = _safe_str(location_val.get("display_name")) if isinstance(location_val, dict) else _safe_str(location_val)

                redirect_url = _safe_str(job.get("redirect_url"))

                normalized_jobs.append(
                    {"title": title, "company": company, "location": loc, "redirect_url": redirect_url}
                )

            _set_cached(what, location, income_type, normalized_jobs)
            return normalized_jobs

    except Exception as e:
        print(f"[DEBUG] fetch_jobs failed: what='{what}' where='{location}' income='{income_type}': {e}")
        return []