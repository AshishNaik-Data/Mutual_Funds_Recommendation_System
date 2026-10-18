"""Live NAV lookups from api.mfapi.in with an in-memory TTL cache.

The notebook fetched all ~625 NAVs up front. Here only the funds being recommended are
looked up (in parallel), results are cached for a few hours (NAVs change once a day),
and any failure falls back to the NAV stored in the dataset.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

import requests

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NavQuote:
    nav: float
    date: str | None  # ISO date of the NAV


class NavClient:
    def __init__(
        self,
        base_url: str = "https://api.mfapi.in/mf",
        *,
        ttl_seconds: int = 6 * 3600,
        timeout: float = 8.0,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.ttl = ttl_seconds
        self.timeout = timeout
        self.session = session or requests.Session()
        self._cache: dict[int, tuple[float, NavQuote | None]] = {}
        self._lock = threading.Lock()

    def latest(self, scheme_code: int) -> NavQuote | None:
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(scheme_code)
            if hit and now - hit[0] < self.ttl:
                return hit[1]
        quote = self._fetch(scheme_code)
        with self._lock:
            # Cache misses briefly too, so a dead scheme code isn't retried on every request.
            self._cache[scheme_code] = (now if quote else now - self.ttl + 300, quote)
        return quote

    def latest_many(self, scheme_codes: list[int]) -> dict[int, NavQuote | None]:
        if not scheme_codes:
            return {}
        with ThreadPoolExecutor(max_workers=min(8, len(scheme_codes))) as pool:
            return dict(zip(scheme_codes, pool.map(self.latest, scheme_codes), strict=True))

    def _fetch(self, scheme_code: int) -> NavQuote | None:
        try:
            resp = self.session.get(f"{self.base_url}/{scheme_code}", timeout=self.timeout)
            if resp.status_code != 200:
                return None
            rows = resp.json().get("data") or []
            if not rows:
                return None
            nav = float(rows[0]["nav"])
            if nav <= 0:
                return None
            return NavQuote(nav=nav, date=_iso_date(rows[0].get("date")))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            log.warning("NAV lookup failed for %s: %s", scheme_code, exc)
            return None


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%d-%m-%Y").date().isoformat()
    except ValueError:
        return value
