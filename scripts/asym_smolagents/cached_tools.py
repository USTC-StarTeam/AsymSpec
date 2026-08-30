"""On-disk cached wrappers for smolagents DuckDuckGoSearchTool + VisitWebpageTool.

Cache is keyed by exact input string (query for search, url for visit_webpage)
and persisted to SQLite at:
    .cache/agent_tool_cache/tool_cache.sqlite

Purpose: agent runs are non-deterministic when search/visit tools hit live web
(snippets drift between runs, content-as-served changes). Caching gives
bit-identical tool outputs across runs of the same mode, making cross-cell
comparison scientifically valid.

Concurrent safe via SQLite WAL mode + module-level lock for write paths.
"""
import hashlib
import os
import sqlite3
import threading
from pathlib import Path

from smolagents import DuckDuckGoSearchTool, VisitWebpageTool


# Project-root .cache (matches existing .cache/ conventions in this repo)
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]  # specsteer-baseline/
CACHE_DIR = _REPO_ROOT / ".cache" / "agent_tool_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "tool_cache.sqlite"

# Override via env if multiple runs want isolated caches
if os.environ.get("AGENT_TOOL_CACHE_DB"):
    CACHE_DB = Path(os.environ["AGENT_TOOL_CACHE_DB"])
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)

_db_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(str(CACHE_DB), timeout=30, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;")
    return c


def _init():
    with _db_lock:
        with _conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS tool_cache (
                    tool TEXT NOT NULL,
                    key  TEXT NOT NULL,
                    raw_input TEXT NOT NULL,
                    value TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    PRIMARY KEY (tool, key)
                )
            """)
            c.commit()


_init()


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_get(tool: str, query: str):
    h = _hash(query)
    with _db_lock:
        with _conn() as c:
            r = c.execute(
                "SELECT value FROM tool_cache WHERE tool=? AND key=?",
                (tool, h),
            ).fetchone()
    return r[0] if r else None


def cache_put(tool: str, query: str, value: str):
    h = _hash(query)
    with _db_lock:
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO tool_cache(tool,key,raw_input,value,ts) "
                "VALUES (?,?,?,?,strftime('%s','now'))",
                (tool, h, query, value),
            )
            c.commit()


def cache_stats() -> dict:
    """Return per-tool {count, hit_size} for diagnostics."""
    with _conn() as c:
        rows = c.execute(
            "SELECT tool, COUNT(*), SUM(LENGTH(value)) FROM tool_cache GROUP BY tool"
        ).fetchall()
    return {r[0]: {"count": r[1], "total_bytes": r[2] or 0} for r in rows}


class CachedDuckDuckGoSearchTool(DuckDuckGoSearchTool):
    """Same interface + name as DuckDuckGoSearchTool; transparent caching."""

    # Keep parent's name="web_search" so agent's tool-use scaffolding works unchanged.

    def forward(self, query: str) -> str:
        hit = cache_get("ddgs", query)
        if hit is not None:
            return hit
        # Cache miss: hit live API, store on success only
        result = super().forward(query)
        cache_put("ddgs", query, result)
        return result


class CachedVisitWebpageTool(VisitWebpageTool):
    """Same interface + name as VisitWebpageTool; transparent caching."""

    def forward(self, url: str) -> str:
        hit = cache_get("visit", url)
        if hit is not None:
            return hit
        result = super().forward(url)
        cache_put("visit", url, result)
        return result


if __name__ == "__main__":
    # Smoke test: cache hit/miss
    import sys, time
    t0 = time.perf_counter()
    tool = CachedDuckDuckGoSearchTool()
    q = sys.argv[1] if len(sys.argv) > 1 else "Battle of Hastings year"
    print(f"[1st call] query={q!r}")
    r1 = tool(q)  # tool.__call__ → forward
    t1 = time.perf_counter() - t0
    print(f"  → result: {r1[:120]!r}")
    print(f"  → took: {t1*1000:.0f}ms")
    t0 = time.perf_counter()
    print(f"[2nd call, should be cache hit]")
    r2 = tool(q)
    t2 = time.perf_counter() - t0
    print(f"  → bit-identical: {r1 == r2}")
    print(f"  → took: {t2*1000:.0f}ms (should be ~10ms if cached)")
    print(f"\nCache stats: {cache_stats()}")
