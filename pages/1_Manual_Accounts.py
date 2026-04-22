"""Screen 1 -- Manual Account Data Editing.

Search -> Browse -> Edit workflow for MANUAL_ACC_DATA_CHANGES.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from utils.categories import (
    get_cat_for_subcat,
    get_grouped_subcats,
    get_subcat_descriptions,
)
from utils.db import fetch_manual_acc_data, is_db_configured, render_connection_debug, save_manual_acc_record
from utils.mock_data import get_manual_acc_data
from utils.styles import page_header, section_header
from utils.validators import validate_iban, validate_ico, validate_rc

PT_TP_OPTIONS = ["PO", "FOP", "FO"]
DB_MODE = is_db_configured()

def _init_form_state():
    defaults = {
        "w_iban": "",
        "w_uni_pt_key": "",
        "w_pt_tp_id": "PO",
        "w_ico": "",
        "w_rc": "",
        "w_party_subcat": "unclassified_general",
        "w_party_validity": 99,
        "w_purpose_subcat": "unclassified_general",
        "w_purpose_validity": 99,
        "last_sel_idx": None,
        "acc_table_ver": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_table_selection():
    """Bump the table key version to force a fresh widget with empty selection."""
    old_key = f"acc_table_{st.session_state['acc_table_ver']}"
    if old_key in st.session_state:
        del st.session_state[old_key]
    st.session_state["acc_table_ver"] += 1


def _load_row_into_form(row: pd.Series):
    st.session_state["w_iban"] = str(row.get("IBAN", ""))
    st.session_state["w_uni_pt_key"] = str(row.get("UNI_PT_KEY", ""))
    st.session_state["w_pt_tp_id"] = str(row.get("PT_TP_ID", "PO"))
    st.session_state["w_ico"] = str(row.get("ICO_NUM", ""))
    st.session_state["w_rc"] = str(row.get("RC_NUM", ""))
    st.session_state["w_party_subcat"] = str(row.get("PARTY_SUBCAT", "unclassified_general"))
    st.session_state["w_party_validity"] = int(row.get("PARTY_SUBCAT_VALIDITY", 99))
    st.session_state["w_purpose_subcat"] = str(row.get("PURPOSE_SUBCAT", "unclassified_general"))
    st.session_state["w_purpose_validity"] = int(row.get("PURPOSE_SUBCAT_VALIDITY", 99))


def _clear_form():
    st.session_state["w_iban"] = ""
    st.session_state["w_uni_pt_key"] = ""
    st.session_state["w_pt_tp_id"] = "PO"
    st.session_state["w_ico"] = ""
    st.session_state["w_rc"] = ""
    st.session_state["w_party_subcat"] = "unclassified_general"
    st.session_state["w_party_validity"] = 99
    st.session_state["w_purpose_subcat"] = "unclassified_general"
    st.session_state["w_purpose_validity"] = 99



_init_form_state()
df = fetch_manual_acc_data() if DB_MODE else get_manual_acc_data()
subcats = get_grouped_subcats()
descs = get_subcat_descriptions()

page_header("Manual Accounts", "Search, create, and edit manual account data entries")

render_connection_debug(["manual_acc"])

# Metrics
with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", len(df))
    m2.metric("PO Accounts", len(df[df["PT_TP_ID"] == "PO"]))
    m3.metric("FOP Accounts", len(df[df["PT_TP_ID"] == "FOP"]))
    m4.metric("FO Accounts", len(df[df["PT_TP_ID"] == "FO"]))

st.markdown("<br>", unsafe_allow_html=True)

left_col, right_col = st.columns([55, 45], gap="large")

with left_col:
    section_header("1. Search & Select")

    with st.container(border=True):
        sc1, sc2, sc3 = st.columns(3)
        search_iban = sc1.text_input("Filter by IBAN", placeholder="e.g. CZ...")
        search_ico = sc2.text_input("Filter by ICO", placeholder="8 digits")
        search_rc = sc3.text_input("Filter by RC", placeholder="9-10 digits")

    if DB_MODE:
        filtered = fetch_manual_acc_data(search_iban, search_ico, search_rc)
    else:
        filtered = df.copy()
        if search_iban and search_iban.strip():
            filtered = filtered[filtered["IBAN"].str.upper().str.contains(search_iban.strip().upper(), na=False)]
        if search_ico and search_ico.strip():
            filtered = filtered[filtered["ICO_NUM"].str.contains(search_ico.strip(), na=False)]
        if search_rc and search_rc.strip():
            filtered = filtered[filtered["RC_NUM"].str.contains(search_rc.strip(), na=False)]

    st.markdown(f"**Results ({len(filtered)} records)**")
    st.caption("Use the checkboxes on the left to select a row for editing")
    display_cols = [
        "IBAN", "UNI_PT_KEY", "PT_TP_ID", "ICO_NUM", "RC_NUM",
        "PARTY_SUBCAT", "PARTY_CAT", "PURPOSE_SUBCAT", "PURPOSE_CAT",
    ]
    existing_cols = [c for c in display_cols if c in filtered.columns]

    # Cache the dataframe to prevent st.dataframe from losing selection on rerun
    display_df = filtered[existing_cols].reset_index(drop=True)
    if "last_display_df" not in st.session_state or not st.session_state.last_display_df.equals(display_df):
        st.session_state.last_display_df = display_df

    # Render dataframe FIRST to guarantee accurate selection state across all interactions
    num_rows = len(st.session_state.last_display_df)
    dynamic_height = min(600, 38 + max(1, num_rows) * 35)

    TABLE_KEY = f"acc_table_{st.session_state['acc_table_ver']}"
    table_event = st.dataframe(
        st.session_state.last_display_df,
        use_container_width=True,
        hide_index=True,
        height=dynamic_height,
        on_select="rerun",
        selection_mode="single-row",
        key=TABLE_KEY,
    )

    sel_rows = table_event.selection.rows
    if sel_rows and len(sel_rows) > 0 and sel_rows[0] < len(filtered):
        current_sel_idx = filtered.index[sel_rows[0]]
    else:
        current_sel_idx = None

    # Update form state if selection changed
    if st.session_state["last_sel_idx"] != current_sel_idx:
        st.session_state["last_sel_idx"] = current_sel_idx
        if current_sel_idx is not None:
            _load_row_into_form(filtered.loc[current_sel_idx])
        else:
            _clear_form()

with right_col:
    section_header("2. Account Details")

    with st.container(border=True):
        if current_sel_idx is not None:
            st.info("**Editing selected account.** Unselect the row to create a new entry.")
        else:
            st.success("**Creating new account.** Fill the form and save.")

        st.markdown("##### Account Identity")
        c1, c2 = st.columns(2)
        c1.text_input("IBAN *", key="w_iban", placeholder="CZ...")
        c2.text_input("UNI_PT_KEY *", key="w_uni_pt_key")

        c3, c4, c5 = st.columns(3)
        pt_tp = c3.selectbox("PT_TP_ID *", options=PT_TP_OPTIONS, key="w_pt_tp_id")

        ico_enabled = pt_tp in ("PO", "FOP")
        rc_enabled = pt_tp in ("FO", "FOP")

        c4.text_input(
            "ICO_NUM" + (" *" if ico_enabled else ""),
            key="w_ico",
            disabled=not ico_enabled,
            placeholder="8 digits" if ico_enabled else "N/A"
        )
        c5.text_input(
            "RC_NUM" + (" *" if rc_enabled else ""),
            key="w_rc",
            disabled=not rc_enabled,
            placeholder="9-10 digits" if rc_enabled else "N/A"
        )

        st.divider()
        st.markdown("##### Party Classification")

        # With key=, value lives in session_state; do not also pass index= (Streamlit warns).
        if st.session_state.get("w_party_subcat") not in subcats:
            st.session_state["w_party_subcat"] = "unclassified_general"

        party_sub = st.selectbox(
            "PARTY_SUBCAT *",
            options=subcats,
            key="w_party_subcat",
            #help=descs.get(st.session_state["w_party_subcat"], ""),
        )
        party_cat = get_cat_for_subcat(party_sub)
        pc1, pc2 = st.columns([2, 1])
        pc1.text_input("PARTY_CAT (auto)", value=party_cat, disabled=True)
        st.caption(descs.get(party_sub, ""))

        is_party_unc = party_sub == "unclassified_general"
        pc2.number_input(
            "Validity (1-99)",
            key="w_party_validity",
            min_value=1,
            max_value=99,
            disabled=is_party_unc
        )

        st.divider()
        st.markdown("##### Purpose Classification")

        if st.session_state.get("w_purpose_subcat") not in subcats:
            st.session_state["w_purpose_subcat"] = "unclassified_general"

        purp_sub = st.selectbox(
            "PURPOSE_SUBCAT *",
            options=subcats,
            key="w_purpose_subcat",
            #help=descs.get(st.session_state["w_purpose_subcat"], ""),
        )
        purp_cat = get_cat_for_subcat(purp_sub)
        pu1, pu2 = st.columns([2, 1])
        pu1.text_input("PURPOSE_CAT (auto)", value=purp_cat, disabled=True)
        st.caption(descs.get(purp_sub, ""))

        is_purp_unc = purp_sub == "unclassified_general"
        pu2.number_input(
            "Validity (1-99) ",
            key="w_purpose_validity",
            min_value=1,
            max_value=99,
            disabled=is_purp_unc
        )

        st.markdown("<br>", unsafe_allow_html=True)

        def on_clear_btn():
            _reset_table_selection()
            st.session_state["last_sel_idx"] = None
            _clear_form()

        def on_save_btn():
            errors = []
            v_iban, m_iban = validate_iban(st.session_state.get("w_iban", ""))
            if not v_iban: errors.append(f"IBAN: {m_iban}")

            pt_tp_id = st.session_state.get("w_pt_tp_id", "PO")
            _ico_enabled = pt_tp_id in ("PO", "FOP")
            _rc_enabled = pt_tp_id in ("FO", "FOP")

            if _ico_enabled:
                v_ico, m_ico = validate_ico(st.session_state.get("w_ico", ""))
                if not v_ico: errors.append(f"ICO: {m_ico}")
            if _rc_enabled:
                v_rc, m_rc = validate_rc(st.session_state.get("w_rc", ""))
                if not v_rc: errors.append(f"RC: {m_rc}")

            w_uni_pt_key_str = str(st.session_state.get("w_uni_pt_key", "")).strip()
            w_uni_pt_key = 0
            if w_uni_pt_key_str.isdigit():
                w_uni_pt_key = int(w_uni_pt_key_str)

            if w_uni_pt_key <= 0:
                errors.append("UNI_PT_KEY must be a positive integer")

            if errors:
                st.session_state["form_errors"] = errors
            else:
                st.session_state["form_errors"] = []
                party_sub = st.session_state.get("w_party_subcat", "unclassified_general")
                purp_sub = st.session_state.get("w_purpose_subcat", "unclassified_general")

                _is_party_unc = party_sub == "unclassified_general"
                _is_purp_unc = purp_sub == "unclassified_general"

                party_cat = get_cat_for_subcat(party_sub)
                purp_cat = get_cat_for_subcat(purp_sub)

                new_row = {
                    "IBAN": st.session_state.get("w_iban", "").strip().upper(),
                    "UNI_PT_KEY": int(w_uni_pt_key),
                    "PT_TP_ID": pt_tp_id,
                    "ICO_NUM": st.session_state.get("w_ico", "").strip() if _ico_enabled else "",
                    "RC_NUM": st.session_state.get("w_rc", "").strip() if _rc_enabled else "",
                    "PARTY_SUBCAT": party_sub,
                    "PARTY_CAT": party_cat,
                    "PARTY_SUBCAT_VALIDITY": 99 if _is_party_unc else int(st.session_state.get("w_party_validity", 99)),
                    "PURPOSE_SUBCAT": purp_sub,
                    "PURPOSE_CAT": purp_cat,
                    "PURPOSE_SUBCAT_VALIDITY": 99 if _is_purp_unc else int(st.session_state.get("w_purpose_validity", 99)),
                    "CREATED_BY": "user",
                    "CREATED_AT": datetime.now(),
                    "UPDATED_AT": datetime.now(),
                }

                idx = st.session_state.get("last_sel_idx")
                if DB_MODE:
                    save_manual_acc_record(new_row)
                    st.session_state["manual_acc_data"] = fetch_manual_acc_data(
                        search_iban,
                        search_ico,
                        search_rc,
                    )
                    st.session_state["form_success"] = (
                        "Record updated successfully!" if idx is not None else "New record created successfully!"
                    )
                    if idx is None:
                        _reset_table_selection()
                        st.session_state["last_sel_idx"] = None
                        _clear_form()
                else:
                    current_df = get_manual_acc_data()
                    if idx is not None and idx in current_df.index:
                        for col, val in new_row.items():
                            current_df.at[idx, col] = val
                        current_df.at[idx, "UPDATED_AT"] = datetime.now()
                        st.session_state["manual_acc_data"] = current_df
                        st.session_state["form_success"] = "Record updated successfully!"
                    else:
                        new_df = pd.DataFrame([new_row])
                        st.session_state["manual_acc_data"] = pd.concat([current_df, new_df], ignore_index=True)
                        st.session_state["form_success"] = "New record created successfully!"

                        _reset_table_selection()
                        st.session_state["last_sel_idx"] = None
                        _clear_form()

        b1, b2 = st.columns(2)
        with b1:
            st.button("Save Record", type="primary", use_container_width=True, on_click=on_save_btn)
        with b2:
            st.button("Clear Form", type="secondary", use_container_width=True, on_click=on_clear_btn)

        if st.session_state.get("form_errors"):
            for e in st.session_state["form_errors"]:
                st.error(e, icon="🚨")
            st.session_state["form_errors"] = []

        if st.session_state.get("form_success"):
            st.success(st.session_state["form_success"])
            st.session_state["form_success"] = None
