"""Databricks SQL connection layer for Delta table operations.

Supports two modes:
  - Databricks App runtime: auto-configured via DATABRICKS_HOST / _TOKEN env vars
  - Local development: configure via env vars or .streamlit/secrets.toml

When DB is not configured, the app falls back to in-memory mock data.
"""

import os
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Table configuration — defaults from env, overridable via sidebar
# ---------------------------------------------------------------------------
_DEFAULT_CATALOG = os.getenv("DATABRICKS_CATALOG", "kyuu_demo")
_DEFAULT_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "trn_test")

TABLE_DEFAULTS = {
    "manual_acc": f"{_DEFAULT_CATALOG}.{_DEFAULT_SCHEMA}.manual_acc_data_changes_new",
    "trn_classified": f"{_DEFAULT_CATALOG}.{_DEFAULT_SCHEMA}.trn_classified_12m_new",
    "trn_validation": f"{_DEFAULT_CATALOG}.{_DEFAULT_SCHEMA}.trn_validation",
}


def get_table(key: str) -> str:
    """Return the current fully-qualified table name for *key*.

    Reads from session state (set by the sidebar configurator) with a
    fallback to the env-based default.
    """
    return st.session_state.get(f"tbl_{key}", TABLE_DEFAULTS[key])


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def _streamlit_secrets_file_exists() -> bool:
    """Avoid touching `st.secrets` when no secrets.toml exists (prevents Streamlit UI noise on Databricks Apps)."""
    candidates = (
        Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml",
        Path("/home/app/.streamlit/secrets.toml"),
        Path("/app/python/source_code/.streamlit/secrets.toml"),
    )
    return any(p.is_file() for p in candidates)


def _get_connection_params() -> dict:
    """Resolve connection params from env vars or Streamlit secrets."""
    host = os.getenv("DATABRICKS_HOST", "")
    token = os.getenv("DATABRICKS_TOKEN", "")
    http_path = os.getenv("DATABRICKS_HTTP_PATH", "")

    if not all([host, token, http_path]) and _streamlit_secrets_file_exists():
        try:
            host = host or st.secrets["DATABRICKS_HOST"]
            token = token or st.secrets["DATABRICKS_TOKEN"]
            http_path = http_path or st.secrets["DATABRICKS_HTTP_PATH"]
        except (KeyError, FileNotFoundError, AttributeError):
            pass

    return {
        "server_hostname": host,
        "access_token": token,
        "http_path": http_path,
    }


def is_db_configured() -> bool:
    """True when all required connection parameters are present."""
    params = _get_connection_params()
    return all(v for v in params.values())


def render_connection_debug(table_keys: list[str]) -> None:
    """Render a debug expander showing connection status, table config, and a test query."""
    import os as _os

    db_ok = is_db_configured()
    params = _get_connection_params()

    with st.expander("Connection & Table Debug", expanded=True):
        st.subheader("Connection")
        for env_key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_HTTP_PATH"):
            _set = bool(str(_os.getenv(env_key, "")).strip())
            st.markdown(f"- `{env_key}`: **{'set' if _set else 'missing'}**")
        st.markdown(f"- `secrets.toml` found: **{_streamlit_secrets_file_exists()}**")
        st.markdown(f"- **`is_db_configured()`**: `{db_ok}`")
        if params["server_hostname"]:
            st.markdown(f"- Host: `{params['server_hostname']}`")

        st.subheader("Tables")
        for key in table_keys:
            st.markdown(f"- **{key}**: `{get_table(key)}`")

        st.subheader("Test Query")
        if not db_ok:
            st.warning("DB not configured — skipping test query (using mock data).")
        else:
            test_table = get_table(table_keys[0])
            test_sql = f"SELECT COUNT(*) AS cnt FROM {test_table}"
            try:
                result = _read(test_sql)
                cnt = result["cnt"].iloc[0]
                st.success(f"`{test_table}` — **{cnt}** rows")
            except Exception as exc:
                st.error(f"Query failed on `{test_table}`:\n```\n{exc}\n```")


@contextmanager
def _cursor():
    """Yield a Databricks SQL cursor, closing connection on exit."""
    from databricks import sql as dbsql

    params = _get_connection_params()
    conn = dbsql.connect(**params)
    try:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    finally:
        conn.close()


def _read(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a SELECT and return a DataFrame."""
    with _cursor() as cur:
        cur.execute(sql, params or {})
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _write(sql: str, params: dict | None = None) -> None:
    """Execute a write statement (INSERT, MERGE, etc.)."""
    with _cursor() as cur:
        cur.execute(sql, params or {})


# ---------------------------------------------------------------------------
# Screen 1 — Manual Account Data (MANUAL_ACC_DATA_CHANGES)
# ---------------------------------------------------------------------------

def fetch_manual_acc_data(
    iban: str = "", ico: str = "", rc: str = "",
) -> pd.DataFrame:
    """Query account data with optional partial-match filters."""
    conditions = ["1=1"]
    params: dict = {}

    if iban.strip():
        conditions.append("UPPER(IBAN) LIKE UPPER(%(iban)s)")
        params["iban"] = f"%{iban.strip()}%"
    if ico.strip():
        conditions.append("ICO_NUM LIKE %(ico)s")
        params["ico"] = f"%{ico.strip()}%"
    if rc.strip():
        conditions.append("RC_NUM LIKE %(rc)s")
        params["rc"] = f"%{rc.strip()}%"

    sql = f"""
        SELECT *
        FROM {get_table('manual_acc')}
        WHERE {' AND '.join(conditions)}
        ORDER BY UPDATED_AT DESC
        LIMIT 1000
    """
    return _read(sql, params)


def save_manual_acc_record(record: dict) -> None:
    """MERGE (upsert) a single record keyed by UNI_PT_KEY."""
    sql = f"""
        MERGE INTO {get_table('manual_acc')} AS t
        USING (
            SELECT
                %(IBAN)s                    AS IBAN,
                CAST(%(UNI_PT_KEY)s AS BIGINT) AS UNI_PT_KEY,
                %(PT_TP_ID)s                AS PT_TP_ID,
                %(ICO_NUM)s                 AS ICO_NUM,
                %(RC_NUM)s                  AS RC_NUM,
                %(PARTY_SUBCAT)s            AS PARTY_SUBCAT,
                %(PARTY_CAT)s               AS PARTY_CAT,
                CAST(%(PARTY_SUBCAT_VALIDITY)s AS INT) AS PARTY_SUBCAT_VALIDITY,
                %(PURPOSE_SUBCAT)s          AS PURPOSE_SUBCAT,
                %(PURPOSE_CAT)s             AS PURPOSE_CAT,
                CAST(%(PURPOSE_SUBCAT_VALIDITY)s AS INT) AS PURPOSE_SUBCAT_VALIDITY,
                %(CREATED_BY)s              AS CREATED_BY,
                CAST(%(CREATED_AT)s AS TIMESTAMP) AS CREATED_AT,
                CAST(%(UPDATED_AT)s AS TIMESTAMP) AS UPDATED_AT
        ) AS s
        ON t.UNI_PT_KEY = s.UNI_PT_KEY
        WHEN MATCHED THEN UPDATE SET
            t.IBAN                    = s.IBAN,
            t.PT_TP_ID               = s.PT_TP_ID,
            t.ICO_NUM                = s.ICO_NUM,
            t.RC_NUM                 = s.RC_NUM,
            t.PARTY_SUBCAT           = s.PARTY_SUBCAT,
            t.PARTY_CAT              = s.PARTY_CAT,
            t.PARTY_SUBCAT_VALIDITY  = s.PARTY_SUBCAT_VALIDITY,
            t.PURPOSE_SUBCAT         = s.PURPOSE_SUBCAT,
            t.PURPOSE_CAT            = s.PURPOSE_CAT,
            t.PURPOSE_SUBCAT_VALIDITY = s.PURPOSE_SUBCAT_VALIDITY,
            t.UPDATED_AT             = s.UPDATED_AT
        WHEN NOT MATCHED THEN INSERT (
            IBAN, UNI_PT_KEY, PT_TP_ID, ICO_NUM, RC_NUM,
            PARTY_SUBCAT, PARTY_CAT, PARTY_SUBCAT_VALIDITY,
            PURPOSE_SUBCAT, PURPOSE_CAT, PURPOSE_SUBCAT_VALIDITY,
            CREATED_BY, CREATED_AT, UPDATED_AT
        ) VALUES (
            s.IBAN, s.UNI_PT_KEY, s.PT_TP_ID, s.ICO_NUM, s.RC_NUM,
            s.PARTY_SUBCAT, s.PARTY_CAT, s.PARTY_SUBCAT_VALIDITY,
            s.PURPOSE_SUBCAT, s.PURPOSE_CAT, s.PURPOSE_SUBCAT_VALIDITY,
            s.CREATED_BY, s.CREATED_AT, s.UPDATED_AT
        )
    """
    _write(sql, record)


# ---------------------------------------------------------------------------
# Screen 2 — Transaction Labeling
# ---------------------------------------------------------------------------

def fetch_trn_for_labeling(
    pay_tp: str = "All",
    src_iban: str = "",
    src_ico: str = "",
    src_rc: str = "",
    dest_iban: str = "",
    dest_ico: str = "",
    dest_rc: str = "",
    purpose_subcat: str = "All",
    date_from: date | None = None,
    date_to: date | None = None,
    last_val_date: date | None = None,
    uncertain_only: bool = False,
    num_rows: int = 50,
) -> pd.DataFrame:
    """Fetch random transactions joined with their latest validation record."""
    conditions = ["1=1"]
    params: dict = {}

    if pay_tp != "All":
        conditions.append("t.PAY_TP_ID = %(pay_tp)s")
        params["pay_tp"] = pay_tp

    _text_filters = [
        ("src_iban", src_iban, "t.SRC_IBAN"),
        ("src_ico", src_ico, "t.SRC_ICO_NUM"),
        ("src_rc", src_rc, "t.SRC_RC_NUM"),
        ("dest_iban", dest_iban, "t.DEST_IBAN"),
        ("dest_ico", dest_ico, "t.DEST_ICO_NUM"),
        ("dest_rc", dest_rc, "t.DEST_RC_NUM"),
    ]
    for key, val, col in _text_filters:
        if val and val.strip():
            conditions.append(f"UPPER({col}) LIKE UPPER(%({key})s)")
            params[key] = f"%{val.strip()}%"

    if purpose_subcat != "All":
        conditions.append("t.PURPOSE_SUBCAT = %(purpose_subcat)s")
        params["purpose_subcat"] = purpose_subcat

    if date_from:
        conditions.append("t.SNAP_DATE >= %(date_from)s")
        params["date_from"] = date_from
    if date_to:
        conditions.append("t.SNAP_DATE <= %(date_to)s")
        params["date_to"] = date_to

    if last_val_date:
        conditions.append(
            "(v.LAST_VALIDATED IS NULL "
            "OR CAST(v.LAST_VALIDATED AS DATE) <= %(last_val_date)s)"
        )
        params["last_val_date"] = last_val_date

    if uncertain_only:
        conditions.append(
            "("
            "COALESCE(v.LAST_PURPOSE_SUBCAT, t.PURPOSE_SUBCAT) IS NULL "
            "OR COALESCE(v.LAST_PURPOSE_SUBCAT, t.PURPOSE_SUBCAT) = '' "
            "OR COALESCE(v.LAST_PURPOSE_SUBCAT, t.PURPOSE_SUBCAT) "
            "   IN ('unclassified_general', 'not_determinable')"
            ")"
        )

    params["num_rows"] = num_rows

    sql = f"""
        WITH latest_validations AS (
            SELECT
                ACC_TRN_KEY,
                MAX(VALIDATION_TIME_STAMP)                       AS LAST_VALIDATED,
                MAX_BY(PURPOSE_SUBCAT, VALIDATION_TIME_STAMP)    AS LAST_PURPOSE_SUBCAT
            FROM {get_table('trn_validation')}
            GROUP BY ACC_TRN_KEY
        )
        SELECT
            t.*,
            v.LAST_VALIDATED,
            COALESCE(v.LAST_PURPOSE_SUBCAT, '') AS LAST_PURPOSE_SUBCAT
        FROM {get_table('trn_classified')} t
        LEFT JOIN latest_validations v ON t.ACC_TRN_KEY = v.ACC_TRN_KEY
        WHERE {' AND '.join(conditions)}
        ORDER BY RAND()
        LIMIT %(num_rows)s
    """
    return _read(sql, params)


def fetch_trn_validations(acc_trn_keys: list | None = None) -> pd.DataFrame:
    """Fetch validation history, optionally filtered to specific transaction keys."""
    if acc_trn_keys is not None and len(acc_trn_keys) == 0:
        return pd.DataFrame(
            columns=["ACC_TRN_KEY", "VALIDATION_TIME_STAMP", "USER", "PURPOSE_SUBCAT", "NOTE"]
        )

    sql = f"SELECT * FROM {get_table('trn_validation')}"
    params: dict = {}

    if acc_trn_keys is not None:
        placeholders = ", ".join(f"%(k{i})s" for i in range(len(acc_trn_keys)))
        sql += f" WHERE ACC_TRN_KEY IN ({placeholders})"
        for i, k in enumerate(acc_trn_keys):
            params[f"k{i}"] = k

    sql += " ORDER BY VALIDATION_TIME_STAMP DESC"
    return _read(sql, params)


def save_trn_validations(validations: list[dict]) -> None:
    """Batch-insert validation records into TRN_VALIDATION."""
    if not validations:
        return

    insert_sql = f"""
        INSERT INTO {get_table('trn_validation')}
            (ACC_TRN_KEY, VALIDATION_TIME_STAMP, `USER`, PURPOSE_SUBCAT, NOTE)
        VALUES
            (%(ACC_TRN_KEY)s, %(VALIDATION_TIME_STAMP)s, %(USER)s, %(PURPOSE_SUBCAT)s, %(NOTE)s)
    """
    with _cursor() as cur:
        for v in validations:
            cur.execute(insert_sql, v)
