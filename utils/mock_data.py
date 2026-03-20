"""Mock data generators for all tables.

Data is seeded for reproducibility and stored in Streamlit session state so it
persists across rerenders.
"""

import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from utils.categories import get_all_subcats, get_cat_for_subcat

_RNG = random.Random(42)

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
    "Petr Svoboda",
    "Eva Kralova",
    "ABC Trading s.r.o.",
    "MONETA Money Bank",
    "Ceska pojistovna a.s.",
    "Skoda Auto a.s.",
    "Alza.cz a.s.",
    "Rohlik Group s.r.o.",
    "Bolt Operations CZ",
    "Lidl Ceska republika v.o.s.",
    "T-Mobile Czech Republic a.s.",
    "Vodafone Czech Republic a.s.",
    "Prazska energetika a.s.",
    "Ceska posta s.p.",
]

_SAMPLE_MESSAGES = [
    "Platba za sluzby",
    "Najem brezen 2026",
    "Faktura 2026-0342",
    "Prevod uspor",
    "Vyplata mezd",
    "Nakup - Albert",
    "Online objednavka #8821",
    "Pojistne - auto",
    "Uver - splatka",
    "Dobiti kreditu",
    "Restaurace U Fleku",
    "Benzin - Shell",
    "Letenky Ryanair",
    "Predplatne Spotify",
    "Dar - charita",
    "",
]


def _rand_iban() -> str:
    return _RNG.choice(_SAMPLE_IBANS)


def _rand_ico() -> str:
    return str(_RNG.randint(10000000, 99999999))


def _rand_rc() -> str:
    y = _RNG.randint(50, 99)
    m = _RNG.randint(1, 12)
    d = _RNG.randint(1, 28)
    suffix = _RNG.randint(100, 9999)
    return f"{y:02d}{m:02d}{d:02d}{suffix:04d}"


def _rand_date(start_year: int = 2025, end_year: int = 2026) -> datetime:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=_RNG.randint(0, delta))


def _rand_subcat() -> str:
    return _RNG.choice(get_all_subcats())


def generate_manual_acc_data(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        pt_tp = _RNG.choice(["PO", "FOP", "FO"])
        ico = _rand_ico() if pt_tp in ("PO", "FOP") else ""
        rc = _rand_rc() if pt_tp in ("FO", "FOP") else ""
        party_sub = _rand_subcat()
        purpose_sub = _rand_subcat()
        created = _rand_date()
        rows.append(
            {
                "IBAN": _rand_iban(),
                "UNI_PT_KEY": 100000 + i,
                "PT_TP_ID": pt_tp,
                "ICO_NUM": ico,
                "RC_NUM": rc,
                "PARTY_SUBCAT": party_sub,
                "PARTY_CAT": get_cat_for_subcat(party_sub),
                "PARTY_SUBCAT_VALIDITY": _RNG.randint(50, 99),
                "PURPOSE_SUBCAT": purpose_sub,
                "PURPOSE_CAT": get_cat_for_subcat(purpose_sub),
                "PURPOSE_SUBCAT_VALIDITY": _RNG.randint(50, 99),
                "CREATED_BY": "system",
                "CREATED_AT": created,
                "UPDATED_AT": created + timedelta(days=_RNG.randint(0, 30)),
            }
        )
    return pd.DataFrame(rows)


def generate_trn_classified(n: int = 200) -> pd.DataFrame:
    subcats = get_all_subcats()
    rows = []
    for i in range(n):
        purpose_sub = _RNG.choice(subcats)
        party_sub = _RNG.choice(subcats)
        pay_tp = _RNG.choice(["CR", "DB"])
        rows.append(
            {
                "ACC_TRN_KEY": 900000 + i,
                "SRC_IBAN": _rand_iban(),
                "SRC_RC_NUM": _rand_rc() if _RNG.random() < 0.4 else "",
                "SRC_ICO_NUM": _rand_ico() if _RNG.random() < 0.4 else "",
                "DEST_IBAN": _rand_iban(),
                "DEST_RC_NUM": _rand_rc() if _RNG.random() < 0.3 else "",
                "DEST_ICO_NUM": _rand_ico() if _RNG.random() < 0.3 else "",
                "DEST_BANK_ACC_NAME": _RNG.choice(_SAMPLE_NAMES),
                "PAY_TP_ID": pay_tp,
                "SNAP_DATE": _rand_date().date(),
                "TRN_AMT_LCCY": round(_RNG.uniform(-50000, 50000), 2),
                "TRN_MSG": _RNG.choice(_SAMPLE_MESSAGES),
                "PARTY_SUBCAT": party_sub,
                "PURPOSE_SUBCAT": purpose_sub,
                "PURPOSE_CAT": get_cat_for_subcat(purpose_sub),
            }
        )
    return pd.DataFrame(rows)


def get_manual_acc_data() -> pd.DataFrame:
    """Get MANUAL_ACC_DATA_CHANGES from session state, seeding if needed."""
    if "manual_acc_data" not in st.session_state:
        st.session_state["manual_acc_data"] = generate_manual_acc_data()
    return st.session_state["manual_acc_data"]


def get_trn_classified() -> pd.DataFrame:
    """Get TRN_CLASSIFIED_12M from session state, seeding if needed."""
    if "trn_classified" not in st.session_state:
        st.session_state["trn_classified"] = generate_trn_classified()
    return st.session_state["trn_classified"]


def get_trn_validations() -> pd.DataFrame:
    """Get TRN_VALIDATION table from session state."""
    if "trn_validations" not in st.session_state:
        st.session_state["trn_validations"] = pd.DataFrame(
            columns=["ACC_TRN_KEY", "VALIDATION_TIME_STAMP", "USER", "PURPOSE_SUBCAT", "NOTE"]
        )
    return st.session_state["trn_validations"]
