import json
import os
import asyncio
import re
import contextvars
import mysql.connector
#from .config import log
from config import log

def _load_db_config() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db-config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("db-config.json not found — database name not set")
        return {}
    except Exception as e:
        log.warning(f"db-config.json load failed: {e}")
        return {}

_db_cfg = _load_db_config()
_DB_DEFAULT    = _db_cfg.get("database", "")
_DB_TABLES     = _db_cfg.get("tables", {})   # db_name -> {memory_shortterm, ...}

# Context variable — set per-request so all DB calls in that request use the
# correct database without threading model_key through every call site.
_active_model_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_active_model_key", default=""
)

def set_model_context(model_key: str) -> None:
    """Set the active model key for DB routing in the current async context."""
    _active_model_key.set(model_key or "")

def get_database_for_model(model_key: str | None = None) -> str:
    """Return the database name for a model key from LLM_REGISTRY, or None if not configured."""
    from config import LLM_REGISTRY
    key = model_key or _active_model_key.get()
    if not key:
        return _DB_DEFAULT
    return LLM_REGISTRY.get(key, {}).get("database") or _DB_DEFAULT

def get_tables_for_model(model_key: str | None = None) -> dict:
    """Return the table name map for the active model's database.

    Falls back to the first table set defined, then bare logical names.
    """
    db = get_database_for_model(model_key)
    if db in _DB_TABLES:
        return _DB_TABLES[db]
    # Fallback: use first defined table set (should not normally happen)
    if _DB_TABLES:
        return next(iter(_DB_TABLES.values()))
    return {}

def _connect() -> mysql.connector.MySQLConnection:
    return mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
        database=get_database_for_model(),
    )

def _run_sql(sql: str) -> str:
    # Fresh connection per call — mysql.connector is not thread-safe with a
    # shared connection when multiple asyncio.to_thread calls run concurrently.
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        if cursor.description:
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            if not rows:
                return "(no rows)"
            col_widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
                          for i, c in enumerate(cols)]
            fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
            sep = "-+-".join("-" * w for w in col_widths)
            lines = [fmt.format(*cols), sep]
            for row in rows:
                lines.append(fmt.format(*[str(v) for v in row]))
            return "\n".join(lines)
        else:
            conn.commit()
            return f"OK — rows affected: {cursor.rowcount}"
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        raise exc
    finally:
        cursor.close()
        try:
            conn.close()
        except Exception:
            pass

async def execute_sql(sql: str) -> str:
    return await asyncio.to_thread(_run_sql, sql)

def _fetch_dicts(sql: str) -> list[dict]:
    """Run a SELECT and return rows as list of dicts (no text formatting)."""
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        if not cursor.description:
            return []
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()
        try:
            conn.close()
        except Exception:
            pass

async def fetch_dicts(sql: str) -> list[dict]:
    """Async wrapper for _fetch_dicts — returns list of dicts, pipe-safe."""
    return await asyncio.to_thread(_fetch_dicts, sql)

def _run_insert(sql: str) -> int:
    """Run an INSERT and return lastrowid within the same connection."""
    conn = _connect()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
        return cursor.lastrowid or 0
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        raise exc
    finally:
        cursor.close()
        try:
            conn.close()
        except Exception:
            pass

async def execute_insert(sql: str) -> int:
    """Execute an INSERT and return the new row id (same-connection, avoids LAST_INSERT_ID race)."""
    return await asyncio.to_thread(_run_insert, sql)

def extract_table_names(sql: str) -> list[str]:
    u = sql.upper()
    patterns = [
        r"\bFROM\s+(\w+)",
        r"\bJOIN\s+(\w+)",
        r"\bINTO\s+(\w+)",
        r"\bUPDATE\s+(\w+)",
        r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
        r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)",
        r"\bDELETE\s+FROM\s+(\w+)",
        r"\bTRUNCATE\s+(?:TABLE\s+)?(\w+)",
    ]
    tables = set()
    for p in patterns:
        for m in re.finditer(p, u):
            tables.add(m.group(1).lower())
    return list(tables)