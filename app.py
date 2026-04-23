import streamlit as st

from utils.db import TABLE_DEFAULTS

st.set_page_config(
    page_title="TrnClassification",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — Table Configuration (persisted in session state)
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.expander("Table Configuration", icon=":material/database:"):
        st.caption("Override catalog.schema.table for each role. Defaults come from env vars.")
        for key, label in [
            ("manual_acc", "Manual Accounts (read/write)"),
            ("trn_classified", "TRN Classified (read)"),
            ("trn_validation", "TRN Validation (read/write)"),
        ]:
            ss_key = f"tbl_{key}"
            if ss_key not in st.session_state:
                st.session_state[ss_key] = TABLE_DEFAULTS[key]
            st.text_input(label, key=ss_key)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
pages = {
    "Navigation": [
        st.Page("home.py", title="Home", icon="🏠"),
        st.Page("pages/1_Manual_Accounts.py", title="Manual Accounts"),
        st.Page("pages/2_Transaction_Labeling.py", title="Transaction Labeling"),
    ]
}

pg = st.navigation(pages)
pg.run()
