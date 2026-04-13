"""
Seed mock Delta tables for TrnClassification (`trn-ops`).

Usage on Databricks:
  - Import this file as a workspace notebook, **or**
  - Copy into two cells split at the second `# COMMAND ----------` marker.

Creates / overwrites (Unity Catalog):
  trn_catalog.trn_schema.MANUAL_ACC_DATA_CHANGES
  trn_catalog.trn_schema.ACC_DATA_TAB_PIM
  trn_catalog.trn_schema.TRN_CLASSIFIED_12M
  trn_catalog.trn_schema.TRN_VALIDATION

Column layouts match `utils/mock_data.py` + `utils/db.py` expectations.

Requires the **trn-ops repo layout** on the driver: `utils/categories.py` and
`categories.json` at the repo root (parent of `databricks/`). The script adds
the repo root to `sys.path` so `utils.categories` resolves like the Streamlit app.
If you open only this file in a notebook, `%cd` to the repo root first or attach
the full repo.
"""

# COMMAND ----------

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd()


REPO_ROOT = _repo_root()
_REPO_STR = str(REPO_ROOT)
if _REPO_STR not in sys.path:
    sys.path.insert(0, _REPO_STR)

try:
    from utils.categories import get_all_subcats, get_cat_for_subcat
except ImportError:
    get_all_subcats = None  # type: ignore[assignment]
    get_cat_for_subcat = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Config — edit for your workspace
# ---------------------------------------------------------------------------

CATALOG = "kyuu_demo"
SCHEMA = "trn_test"

N_MANUAL = 12
N_TRN = 200
RNG_SEED = 42

_SAMPLE_IBANS = [
    "CZ4701000000001234567890",
    "CZ1208000000009876543210",
    "CZ4103000000001122334455",
    "CZ3306000000005544332211",
    "CZ9720100000009988776655",
    "CZ1408000000001111111111",
    "CZ6408000000002222222222",
    "CZ5701000000003333333333",
    "CZ1001000000004444444444",
    "CZ7306000000005555555555",
    "SK6702000000001234567890",
    "PL41101000000000000012345678",
]

_SAMPLE_NAMES = [
    "Jan Novak",
    "ABC Trading s.r.o.",
    "MONETA Money Bank",
    "Skoda Auto a.s.",
    "Alza.cz a.s.",
]

_SAMPLE_MESSAGES = [
    "Platba za sluzby",
    "Faktura 2026-0342",
    "Vyplata mezd",
    "",
]

def _load_categories_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.loads(f.read().replace("\u00a0", " "))


def _subcats_from_raw(raw: dict) -> list[str]:
    names = [e["name"] for entries in raw.values() for e in entries]
    return sorted(set(names))


def _subcat_to_cat_from_raw(raw: dict) -> dict[str, str]:
    return {e["name"]: cat for cat, entries in raw.items() for e in entries}


def load_subcats() -> list[str]:
    """Subcategory names: same as `utils.categories.get_all_subcats()`."""
    if get_all_subcats is not None:
        return get_all_subcats()
    path = REPO_ROOT / "categories.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot import utils.categories and categories.json missing at {path}. "
            "Use the full trn-ops repo and run from repo root."
        )
    return _subcats_from_raw(_load_categories_json(path))


def parent_cat_for_subcat(subcat: str) -> str:
    """Parent category: same as `utils.categories.get_cat_for_subcat()`."""
    if get_cat_for_subcat is not None:
        return get_cat_for_subcat(subcat)
    path = REPO_ROOT / "categories.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot import utils.categories and categories.json missing at {path}."
        )
    return _subcat_to_cat_from_raw(_load_categories_json(path)).get(subcat, "")


def _rand_rc(rng: Random) -> str:
    y = rng.randint(50, 99)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    suffix = rng.randint(100, 9999)
    return f"{y:02d}{m:02d}{d:02d}{suffix:04d}"


def _rand_date(rng: Random, start_year: int = 2025, end_year: int = 2026) -> datetime:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, delta_days))


def _build_manual_rows(subcats: list[str], n: int) -> list[tuple]:
    rng = Random(RNG_SEED)
    pt_opts = ["PO", "FOP", "FO"]
    rows: list[tuple] = []
    for i in range(n):
        pt = rng.choice(pt_opts)
        ico = str(rng.randint(10000000, 99999999)) if pt in ("PO", "FOP") else ""
        if pt in ("FO", "FOP"):
            y, m, d = rng.randint(50, 99), rng.randint(1, 12), rng.randint(1, 28)
            suffix = rng.randint(100, 9999)
            rc = f"{y:02d}{m:02d}{d:02d}{suffix:04d}"
        else:
            rc = ""
        party_sub = rng.choice(subcats)
        purpose_sub = rng.choice(subcats)
        created = _rand_date(rng)
        updated = created + timedelta(days=rng.randint(0, 30))
        rows.append(
            (
                rng.choice(_SAMPLE_IBANS),
                int(100000 + i),
                pt,
                ico,
                rc,
                party_sub,
                parent_cat_for_subcat(party_sub),
                int(rng.randint(50, 99)),
                purpose_sub,
                parent_cat_for_subcat(purpose_sub),
                int(rng.randint(50, 99)),
                "system_seed",
                created,
                updated,
            )
        )
    return rows


def _build_trn_rows(subcats: list[str], n: int) -> list[tuple]:
    rng = Random(RNG_SEED + 1)
    rows: list[tuple] = []
    for i in range(n):
        purpose_sub = rng.choice(subcats)
        party_sub = rng.choice(subcats)
        pay_tp = rng.choice(["CR", "DB"])
        src_rc = _rand_rc(rng) if rng.random() < 0.4 else ""
        src_ico = str(rng.randint(10000000, 99999999)) if rng.random() < 0.4 else ""
        dest_rc = _rand_rc(rng) if rng.random() < 0.3 else ""
        dest_ico = str(rng.randint(10000000, 99999999)) if rng.random() < 0.3 else ""
        snap = _rand_date(rng).date()
        rows.append(
            (
                int(900000 + i),
                rng.choice(_SAMPLE_IBANS),
                src_rc,
                src_ico,
                rng.choice(_SAMPLE_IBANS),
                dest_rc,
                dest_ico,
                rng.choice(_SAMPLE_NAMES),
                pay_tp,
                snap,
                round(rng.uniform(-50000, 50000), 2),
                rng.choice(_SAMPLE_MESSAGES),
                party_sub,
                purpose_sub,
                parent_cat_for_subcat(purpose_sub),
            )
        )
    return rows


def _validation_schema() -> StructType:
    return StructType(
        [
            StructField("ACC_TRN_KEY", LongType(), False),
            StructField("VALIDATION_TIME_STAMP", TimestampType(), False),
            StructField("USER", StringType(), False),
            StructField("PURPOSE_SUBCAT", StringType(), False),
            StructField("NOTE", StringType(), True),
        ]
    )


# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE {SCHEMA}")

subcats = load_subcats()

manual_cols = [
    "IBAN",
    "UNI_PT_KEY",
    "PT_TP_ID",
    "ICO_NUM",
    "RC_NUM",
    "PARTY_SUBCAT",
    "PARTY_CAT",
    "PARTY_SUBCAT_VALIDITY",
    "PURPOSE_SUBCAT",
    "PURPOSE_CAT",
    "PURPOSE_SUBCAT_VALIDITY",
    "CREATED_BY",
    "CREATED_AT",
    "UPDATED_AT",
]
manual_df = spark.createDataFrame(_build_manual_rows(subcats, N_MANUAL), manual_cols)

# PIM: same columns as manual for lookup / enrichment smoke tests
pim_df = manual_df.withColumn("CREATED_BY", F.lit("pim_seed"))

trn_cols = [
    "ACC_TRN_KEY",
    "SRC_IBAN",
    "SRC_RC_NUM",
    "SRC_ICO_NUM",
    "DEST_IBAN",
    "DEST_RC_NUM",
    "DEST_ICO_NUM",
    "DEST_BANK_ACC_NAME",
    "PAY_TP_ID",
    "SNAP_DATE",
    "TRN_AMT_LCCY",
    "TRN_MSG",
    "PARTY_SUBCAT",
    "PURPOSE_SUBCAT",
    "PURPOSE_CAT",
]
trn_df = spark.createDataFrame(_build_trn_rows(subcats, N_TRN), trn_cols)

val_rows = [
    (
        900000,
        datetime(2026, 1, 10, 12, 0, 0),
        "analyst_seed",
        "unclassified_general",
        "seed row",
    ),
    (
        900001,
        datetime(2026, 2, 15, 8, 30, 0),
        "analyst_seed",
        "charity",
        "",
    ),
]
val_df = spark.createDataFrame(val_rows, _validation_schema())

manual_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.MANUAL_ACC_DATA_CHANGES"
)
pim_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ACC_DATA_TAB_PIM"
)
trn_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.TRN_CLASSIFIED_12M"
)
val_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.TRN_VALIDATION"
)

print("Done. Tables:")
for t in (
    "MANUAL_ACC_DATA_CHANGES",
    "ACC_DATA_TAB_PIM",
    "TRN_CLASSIFIED_12M",
    "TRN_VALIDATION",
):
    print(f"  {CATALOG}.{SCHEMA}.{t}")
