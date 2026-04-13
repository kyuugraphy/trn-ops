"""Screen 2 -- Transaction Labeling.

Guided workflow: Filter -> Review -> Validate -> Save.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

from utils.categories import get_grouped_subcats
from utils.db import (
    fetch_trn_for_labeling,
    fetch_trn_validations as fetch_trn_validations_db,
    is_db_configured,
    save_trn_validations,
)
from utils.mock_data import get_trn_classified, get_trn_validations
from utils.styles import page_header, section_header

_DEFAULT_COLUMNS = [
    "SRC_IBAN",
    "PAY_TP_ID",
    "SNAP_DATE",
    "TRN_AMT_LCCY",
    "DEST_IBAN",
    "DEST_BANK_ACC_NAME",
    "TRN_MSG",
    "PARTY_SUBCAT",
    "PURPOSE_SUBCAT",
]

_ALL_DISPLAY_COLUMNS = [
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
DB_MODE = is_db_configured()


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = df.copy()

    if filters.get("pay_tp") and filters["pay_tp"] != "All":
        result = result[result["PAY_TP_ID"] == filters["pay_tp"]]

    for col_key, col_name in [
        ("src_iban", "SRC_IBAN"),
        ("src_ico", "SRC_ICO_NUM"),
        ("src_rc", "SRC_RC_NUM"),
        ("dest_iban", "DEST_IBAN"),
        ("dest_ico", "DEST_ICO_NUM"),
        ("dest_rc", "DEST_RC_NUM"),
    ]:
        val = filters.get(col_key, "").strip()
        if val:
            result = result[result[col_name].str.contains(val, case=False, na=False)]

    if filters.get("purpose_subcat") and filters["purpose_subcat"] != "All":
        result = result[result["PURPOSE_SUBCAT"] == filters["purpose_subcat"]]

    if filters.get("date_from"):
        result = result[result["SNAP_DATE"] >= filters["date_from"]]
    if filters.get("date_to"):
        result = result[result["SNAP_DATE"] <= filters["date_to"]]

    n_rows = filters.get("num_rows", 50)
    if len(result) > n_rows:
        result = result.sample(n=n_rows, random_state=42)

    return result.reset_index(drop=True)


def _join_with_validations(trn_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Join transactions with their latest validation record."""
    if val_df.empty:
        trn_df["LAST_VALIDATED"] = pd.NaT
        trn_df["LAST_PURPOSE_SUBCAT"] = ""
        return trn_df

    latest = (
        val_df.sort_values("VALIDATION_TIME_STAMP")
        .groupby("ACC_TRN_KEY")
        .last()
        .reset_index()[["ACC_TRN_KEY", "VALIDATION_TIME_STAMP", "PURPOSE_SUBCAT"]]
    )
    latest.columns = ["ACC_TRN_KEY", "LAST_VALIDATED", "LAST_PURPOSE_SUBCAT"]

    merged = trn_df.merge(latest, on="ACC_TRN_KEY", how="left")
    merged["LAST_VALIDATED"] = merged["LAST_VALIDATED"].where(merged["LAST_VALIDATED"].notna(), other=pd.NaT)
    merged["LAST_PURPOSE_SUBCAT"] = merged["LAST_PURPOSE_SUBCAT"].fillna("")
    return merged


def _filter_by_validation_date(df: pd.DataFrame, last_validated_date: date | None) -> pd.DataFrame:
    if last_validated_date is None:
        return df
    mask_never = df["LAST_VALIDATED"].isna()
    mask_before = df["LAST_VALIDATED"].notna() & (
        pd.to_datetime(df["LAST_VALIDATED"]).dt.date <= last_validated_date
    )
    return df[mask_never | mask_before]


def _filter_uncertain(df: pd.DataFrame, show_uncertain: bool) -> pd.DataFrame:
    """When uncertain toggle is ON, keep only rows whose last validation was
    'uncertain' (i.e. last validated PURPOSE_SUBCAT contains 'unclassified').
    """
    if not show_uncertain:
        return df
    return df[df["LAST_PURPOSE_SUBCAT"].str.contains("unclassified", case=False, na=False) | df["LAST_PURPOSE_SUBCAT"].eq("")]


if True:
    subcats = get_grouped_subcats()
    all_subcats_with_extra = subcats + ["not_determinable"]

    page_header("Transaction Labeling", "Review and validate classified transactions")

    # -- Step indicator --
    current_step = st.session_state.get("labeling_step", 0)
    sac.steps(
        items=[
            sac.StepsItem(title="Filter", icon="funnel"),
            sac.StepsItem(title="Review", icon="table"),
            sac.StepsItem(title="Save", icon="check-circle"),
        ],
        index=current_step,
        variant="default",
        dot=False,
    )

    st.markdown("")

    # -- Metrics row --
    loaded_df = st.session_state.get("labeling_data")
    total_loaded = len(loaded_df) if loaded_df is not None else 0
    validated_count = 0
    if loaded_df is not None and "Validated" in loaded_df.columns:
        validated_count = int(loaded_df["Validated"].sum())
    val_table = fetch_trn_validations_db() if DB_MODE else get_trn_validations()

    with st.container(border=True):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Loaded Transactions", total_loaded)
        mc2.metric("Validated (this batch)", validated_count)
        mc3.metric("Pending", max(0, total_loaded - validated_count))
        mc4.metric("Total Saved Validations", len(val_table))

    st.markdown("<br>", unsafe_allow_html=True)

    # ================================================================
    # FILTER PANEL
    # ================================================================
    with st.expander("Filters", expanded=loaded_df is None, icon=":material/filter_alt:"):
        section_header("Source Filters")
        sf1, sf2, sf3 = st.columns(3)
        src_iban = sf1.text_input("SRC_IBAN", key="lbl_src_iban", placeholder="Source IBAN")
        src_ico = sf2.text_input("SRC_ICO_NUM", key="lbl_src_ico", placeholder="8 digits")
        src_rc = sf3.text_input("SRC_RC_NUM", key="lbl_src_rc", placeholder="9-10 digits")

        section_header("Destination Filters")
        df1, df2, df3 = st.columns(3)
        dest_iban = df1.text_input("DEST_IBAN", key="lbl_dest_iban", placeholder="Destination IBAN")
        dest_ico = df2.text_input("DEST_ICO_NUM", key="lbl_dest_ico", placeholder="8 digits")
        dest_rc = df3.text_input("DEST_RC_NUM", key="lbl_dest_rc", placeholder="9-10 digits")

        section_header("Transaction Filters")
        tf1, tf2, tf3, tf4 = st.columns(4)
        pay_tp = tf1.selectbox("PAY_TP_ID", options=["All", "CR", "DB"], key="lbl_pay_tp")
        purpose_filter = tf2.selectbox(
            "PURPOSE_SUBCAT",
            options=["All"] + subcats,
            key="lbl_purpose_filter",
        )
        date_from = tf3.date_input("Date From", value=date(2025, 1, 1), key="lbl_date_from")
        date_to = tf4.date_input("Date To", value=date(2026, 12, 31), key="lbl_date_to")

        tf5, tf6, tf7 = st.columns(3)
        num_rows = tf5.slider("Number of Rows", min_value=10, max_value=500, value=50, step=10, key="lbl_num_rows")
        last_val_date = tf6.date_input(
            "Last Validated Before",
            value=None,
            key="lbl_last_val_date",
            help="Show transactions whose last validation is on or before this date, or never validated.",
        )
        uncertain = tf7.toggle(
            "Show Uncertain Only",
            value=False,
            key="lbl_uncertain",
            help="Show only transactions whose last validation was uncertain (unclassified).",
        )

        section_header("Display Options")
        do1, do2 = st.columns([3, 1])
        selected_columns = do1.multiselect(
            "Visible Columns",
            options=_ALL_DISPLAY_COLUMNS,
            default=_DEFAULT_COLUMNS,
            key="lbl_columns",
        )
        user_name = do2.text_input("Your Name", value="analyst", key="lbl_user", placeholder="User name for validation")

        st.markdown("<br>", unsafe_allow_html=True)
        load_clicked = st.button("Load Transactions", type="primary", key="btn_query", use_container_width=True)

    # ================================================================
    # LOAD DATA
    # ================================================================
    if load_clicked:
        if DB_MODE:
            joined = fetch_trn_for_labeling(
                pay_tp=pay_tp,
                src_iban=src_iban,
                src_ico=src_ico,
                src_rc=src_rc,
                dest_iban=dest_iban,
                dest_ico=dest_ico,
                dest_rc=dest_rc,
                purpose_subcat=purpose_filter,
                date_from=date_from,
                date_to=date_to,
                last_val_date=last_val_date,
                uncertain_only=uncertain,
                num_rows=num_rows,
            )
        else:
            filters = {
                "pay_tp": pay_tp,
                "src_iban": src_iban,
                "src_ico": src_ico,
                "src_rc": src_rc,
                "dest_iban": dest_iban,
                "dest_ico": dest_ico,
                "dest_rc": dest_rc,
                "purpose_subcat": purpose_filter,
                "date_from": date_from,
                "date_to": date_to,
                "num_rows": num_rows,
            }
            raw = get_trn_classified()
            filtered = _apply_filters(raw, filters)
            joined = _join_with_validations(filtered, get_trn_validations())
            joined = _filter_by_validation_date(joined, last_val_date)
            joined = _filter_uncertain(joined, uncertain)

        joined["Validated"] = False
        joined["CORRECTED_PURPOSE_SUBCAT"] = None
        joined["NOTE"] = ""

        st.session_state["labeling_data"] = joined
        st.session_state["labeling_step"] = 1
        st.rerun()

    # ================================================================
    # REVIEW TABLE
    # ================================================================
    labeling_df = st.session_state.get("labeling_data")

    if labeling_df is not None and not labeling_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Transaction Review")

        # Toolbar
        tb1, tb2, tb3 = st.columns([1, 1, 4])
        with tb1:
            validate_all = st.button("Validate All", type="secondary", key="btn_validate_all", use_container_width=True)
        with tb2:
            save_validation = st.button("Save Validation", type="primary", key="btn_save_validation", use_container_width=True)
        with tb3:
            confirmed = labeling_df["Validated"].sum() if "Validated" in labeling_df.columns else 0
            sac.tags(
                [
                    sac.Tag(label=f"{len(labeling_df)} loaded", color="blue"),
                    sac.Tag(label=f"{int(confirmed)} validated", color="green"),
                    sac.Tag(label=f"{len(labeling_df) - int(confirmed)} pending", color="orange"),
                ],
                align="end",
            )

        editable_cols = ["Validated", "CORRECTED_PURPOSE_SUBCAT", "NOTE"]
        visible = ["ACC_TRN_KEY"] + [c for c in selected_columns if c in labeling_df.columns]
        visible = visible + [c for c in editable_cols if c not in visible]
        visible = list(dict.fromkeys(visible))

        column_config = {
            "Validated": st.column_config.CheckboxColumn("Validated", default=False),
            "CORRECTED_PURPOSE_SUBCAT": st.column_config.SelectboxColumn(
                "Corrected Purpose",
                options=all_subcats_with_extra,
                required=False,
            ),
            "NOTE": st.column_config.TextColumn("Note", max_chars=500),
            "TRN_AMT_LCCY": st.column_config.NumberColumn("Amount (CZK)", format="%.2f"),
            "SNAP_DATE": st.column_config.DateColumn("Snap Date"),
            "ACC_TRN_KEY": st.column_config.NumberColumn("TRN Key", disabled=True),
        }

        disabled_cols = [c for c in visible if c not in editable_cols]

        # Calculate dynamic height (approx 35px per row + 38px for header)
        num_rows = len(labeling_df)
        dynamic_height = min(600, 38 + max(1, num_rows) * 35)

        edited = st.data_editor(
            labeling_df[visible],
            column_config=column_config,
            disabled=disabled_cols,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            height=dynamic_height,
            key="labeling_editor",
        )

        if validate_all:
            labeling_df["Validated"] = True
            st.session_state["labeling_data"] = labeling_df
            st.session_state["labeling_step"] = 2
            st.toast("All transactions marked as validated!")
            st.rerun()

        if edited is not None:
            for col in editable_cols:
                if col in edited.columns:
                    labeling_df[col] = edited[col].values
            st.session_state["labeling_data"] = labeling_df

        if save_validation:
            validated_rows = labeling_df[labeling_df["Validated"] == True]  # noqa: E712

            if validated_rows.empty:
                sac.alert(
                    label="No rows are marked as validated. Please validate at least one row before saving.",
                    icon="exclamation-triangle",
                    color="warning",
                )
            else:
                user = st.session_state.get("lbl_user", "analyst") or "analyst"
                now = datetime.now()

                new_validations = []
                for _, row in validated_rows.iterrows():
                    corrected = row.get("CORRECTED_PURPOSE_SUBCAT")
                    original = row.get("PURPOSE_SUBCAT", "")
                    final_purpose = corrected if (corrected and pd.notna(corrected)) else original

                    new_validations.append(
                        {
                            "ACC_TRN_KEY": row["ACC_TRN_KEY"],
                            "VALIDATION_TIME_STAMP": now,
                            "USER": user,
                            "PURPOSE_SUBCAT": final_purpose,
                            "NOTE": row.get("NOTE", ""),
                        }
                    )

                if DB_MODE:
                    save_trn_validations(new_validations)
                else:
                    new_val_df = pd.DataFrame(new_validations)
                    existing = get_trn_validations()
                    st.session_state["trn_validations"] = pd.concat(
                        [existing, new_val_df], ignore_index=True
                    )

                st.session_state["labeling_step"] = 2
                st.session_state["labeling_data"] = None
                st.toast(f"Saved {len(new_validations)} validation(s) successfully!")
                st.rerun()

    elif labeling_df is not None and labeling_df.empty:
        sac.alert(
            label="No transactions match the selected filters. Try adjusting your criteria.",
            icon="info-circle",
            color="info",
        )
