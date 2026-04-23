"""Screen 2 -- Transaction Labeling.

Guided workflow: Filter -> Review -> Validate -> Save.
"""

from datetime import date, datetime

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

from utils.categories import get_grouped_subcats
from utils.db import (
    fetch_trn_for_labeling,
    fetch_trn_validations as fetch_trn_validations_db,
    get_current_user,
    is_db_configured,
    render_connection_debug,
    save_trn_validations,
)
from utils.mock_data import get_trn_classified, get_trn_validations
from utils.styles import inject_custom_css, page_header, section_header

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
    "LAST_VALIDATED",
    "LAST_VALIDATED_BY",
    "LAST_PURPOSE_SUBCAT",
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
    "LAST_VALIDATED",
    "LAST_VALIDATED_BY",
    "LAST_PURPOSE_SUBCAT",
]


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
        trn_df["LAST_VALIDATED_BY"] = ""
        trn_df["LAST_PURPOSE_SUBCAT"] = ""
        return trn_df

    latest = (
        val_df.sort_values("VALIDATION_TIME_STAMP")
        .groupby("ACC_TRN_KEY")
        .last()
        .reset_index()[["ACC_TRN_KEY", "VALIDATION_TIME_STAMP", "USER", "PURPOSE_SUBCAT"]]
    )
    latest.columns = ["ACC_TRN_KEY", "LAST_VALIDATED", "LAST_VALIDATED_BY", "LAST_PURPOSE_SUBCAT"]

    merged = trn_df.merge(latest, on="ACC_TRN_KEY", how="left")
    merged["LAST_VALIDATED"] = merged["LAST_VALIDATED"].where(merged["LAST_VALIDATED"].notna(), other=pd.NaT)
    merged["LAST_VALIDATED_BY"] = merged["LAST_VALIDATED_BY"].fillna("")
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


def _filter_uncertain(
    df: pd.DataFrame,
    show_uncertain: bool,
    last_validated_date: date | None = None,
) -> pd.DataFrame:
    """When uncertain toggle is ON, keep only rows whose PURPOSE_SUBCAT is
    None/empty, 'unclassified_general', or 'not_determinable'.

    If the transaction has a prior validation within the *last_validated_date*
    window, the validated PURPOSE_SUBCAT takes precedence over the original.
    """
    if not show_uncertain:
        return df

    _UNCERTAIN_VALUES = {"unclassified_general", "not_determinable"}

    has_relevant_validation = df["LAST_PURPOSE_SUBCAT"].ne("") & df["LAST_PURPOSE_SUBCAT"].notna()
    if last_validated_date is not None:
        has_relevant_validation = has_relevant_validation & (
            pd.to_datetime(df["LAST_VALIDATED"]).dt.date <= last_validated_date
        )

    effective_purpose = df["PURPOSE_SUBCAT"].copy()
    effective_purpose[has_relevant_validation] = df.loc[has_relevant_validation, "LAST_PURPOSE_SUBCAT"]

    is_null = effective_purpose.isna() | effective_purpose.eq("")
    is_uncertain = effective_purpose.isin(_UNCERTAIN_VALUES)

    return df[is_null | is_uncertain]


# Re-evaluate each run so env changes after initial import are picked up.
DB_MODE = is_db_configured()
subcats = get_grouped_subcats()
all_subcats_with_extra = subcats + ["not_determinable"]

inject_custom_css()
page_header("Transaction Labeling", "Review and validate classified transactions")

render_connection_debug(["trn_classified", "trn_validation"])

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
    editor_state = st.session_state.get("labeling_editor")
    if editor_state is not None and "edited_rows" in editor_state:
        _tmp = loaded_df["Validated"].copy()
        for row_idx_str, changes in editor_state["edited_rows"].items():
            if "Validated" in changes:
                _tmp.iloc[int(row_idx_str)] = changes["Validated"]
        validated_count = int(_tmp.sum())
    else:
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
    date_from = tf3.date_input(
        "Date From",
        value=None,
        key="lbl_date_from",
        help="Leave empty for no lower bound.",
    )
    date_to = tf4.date_input(
        "Date To",
        value=None,
        key="lbl_date_to",
        help="Leave empty for no upper bound.",
    )

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
        help="Show only transactions whose PURPOSE_SUBCAT is null, unclassified_general, or not_determinable (considers last validation when available).",
    )

    section_header("Display Options")
    do1, do2 = st.columns([3, 1])
    selected_columns = do1.multiselect(
        "Visible Columns",
        options=_ALL_DISPLAY_COLUMNS,
        default=_DEFAULT_COLUMNS,
        key="lbl_columns",
    )
    current_user = get_current_user()
    do2.text_input(
        "Signed in as",
        value=current_user,
        disabled=True,
        key="lbl_user_display",
        help="Authenticated Databricks user. Used to attribute saved validations.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    load_clicked = st.button("Load Transactions", type="primary", key="btn_query", use_container_width=True)

# ================================================================
# LOAD DATA
# ================================================================
if load_clicked:
    st.session_state.pop("load_error", None)
    if DB_MODE:
        try:
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
        except Exception as exc:
            st.session_state["load_error"] = str(exc)
            joined = pd.DataFrame()
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
        joined = _filter_uncertain(joined, uncertain, last_val_date)

    joined["Validated"] = False
    joined["CORRECTED_PURPOSE_SUBCAT"] = pd.Series([None] * len(joined), dtype="object")
    joined["NOTE"] = ""

    st.session_state["labeling_data"] = joined
    st.session_state["labeling_step"] = 1
    st.rerun()

# ================================================================
# REVIEW TABLE
# ================================================================
if st.session_state.get("load_error"):
    st.error(f"Failed to load transactions:\n```\n{st.session_state['load_error']}\n```")

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
        sac.tags(
            [
                sac.Tag(label=f"{len(labeling_df)} loaded", color="blue"),
                sac.Tag(label=f"{validated_count} validated", color="green"),
                sac.Tag(label=f"{total_loaded - validated_count} pending", color="orange"),
            ],
            align="end",
        )

    editable_cols = ["Validated", "CORRECTED_PURPOSE_SUBCAT", "NOTE"]

    # Preferred left-to-right column order. Any columns selected by the user
    # that aren't explicitly listed here are appended afterwards in their
    # original selection order.
    _PRIORITY_ORDER = [
        "ACC_TRN_KEY",
        "TRN_MSG",
        "PURPOSE_SUBCAT",
        "LAST_VALIDATED",
        "LAST_VALIDATED_BY",
        "LAST_PURPOSE_SUBCAT",
        "Validated",
        "CORRECTED_PURPOSE_SUBCAT",
        "NOTE",
    ]

    candidate_cols = ["ACC_TRN_KEY"] + list(selected_columns) + editable_cols
    candidate_cols = [c for c in candidate_cols if c in labeling_df.columns or c in editable_cols]

    ordered = [c for c in _PRIORITY_ORDER if c in candidate_cols]
    ordered += [c for c in candidate_cols if c not in ordered]
    visible = list(dict.fromkeys(ordered))

    # Extend selectbox options with any values present in the loaded data so
    # values from the DB that aren't in categories.json (e.g. legacy or custom
    # subcategories) are still renderable by the SelectboxColumn.
    _observed = set()
    for _col in ("PURPOSE_SUBCAT", "LAST_PURPOSE_SUBCAT", "CORRECTED_PURPOSE_SUBCAT"):
        if _col in labeling_df.columns:
            _observed.update(
                str(v) for v in labeling_df[_col].dropna().unique() if str(v).strip()
            )
    _extra_opts = sorted(_observed - set(all_subcats_with_extra))
    purpose_options = all_subcats_with_extra + _extra_opts

    column_config = {
        "Validated": st.column_config.CheckboxColumn("Validated", default=False),
        "CORRECTED_PURPOSE_SUBCAT": st.column_config.SelectboxColumn(
            "Corrected Purpose",
            options=purpose_options,
            required=False,
        ),
        "NOTE": st.column_config.TextColumn("Note", max_chars=500),
        "TRN_AMT_LCCY": st.column_config.NumberColumn("Amount (CZK)", format="%.2f"),
        "SNAP_DATE": st.column_config.DateColumn("Snap Date"),
        "ACC_TRN_KEY": st.column_config.NumberColumn("TRN Key", disabled=True),
        "LAST_VALIDATED": st.column_config.DateColumn(
            "Last Validated", format="YYYY-MM-DD", disabled=True
        ),
        "LAST_VALIDATED_BY": st.column_config.TextColumn("Validated By", disabled=True),
        "LAST_PURPOSE_SUBCAT": st.column_config.TextColumn("Last Val. Purpose", disabled=True),
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

    # -----------------------------------------------------------
    # Auto-fill CORRECTED_PURPOSE_SUBCAT when Validated is ticked
    # -----------------------------------------------------------
    def _pick_default_purpose(row: pd.Series) -> str | None:
        """Prefer the last validated purpose, fall back to the classified one."""
        for col in ("LAST_PURPOSE_SUBCAT", "PURPOSE_SUBCAT"):
            if col not in row.index:
                continue
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return str(val)
        return None

    _needs_rerun = False
    if edited is not None:
        for idx in edited.index:
            row_validated = edited.at[idx, "Validated"]
            was_validated = labeling_df.at[idx, "Validated"]
            corrected_val = edited.at[idx, "CORRECTED_PURPOSE_SUBCAT"]
            corrected_empty = (pd.isna(corrected_val) or corrected_val == "" or corrected_val is None)

            if row_validated and not was_validated and corrected_empty:
                default_purpose = _pick_default_purpose(labeling_df.loc[idx])
                labeling_df.at[idx, "CORRECTED_PURPOSE_SUBCAT"] = default_purpose
                labeling_df.at[idx, "Validated"] = True
                _needs_rerun = True
            elif row_validated != was_validated:
                labeling_df.at[idx, "Validated"] = row_validated

    if _needs_rerun:
        st.session_state["labeling_data"] = labeling_df
        if "labeling_editor" in st.session_state:
            del st.session_state["labeling_editor"]
        st.rerun()

    if validate_all:
        for idx in labeling_df.index:
            corrected_val = labeling_df.at[idx, "CORRECTED_PURPOSE_SUBCAT"]
            if pd.isna(corrected_val) or corrected_val == "" or corrected_val is None:
                labeling_df.at[idx, "CORRECTED_PURPOSE_SUBCAT"] = _pick_default_purpose(
                    labeling_df.loc[idx]
                )
        labeling_df["Validated"] = True
        st.session_state["labeling_data"] = labeling_df
        st.session_state["labeling_step"] = 2
        if "labeling_editor" in st.session_state:
            del st.session_state["labeling_editor"]
        st.toast("All transactions marked as validated!")
        st.rerun()

    # Merge user edits into the working df ONLY for reads (save, metrics),
    # but do NOT write back into labeling_data — that would change the
    # data_editor source and reset the widget on the next rerun.
    working_df = labeling_df.copy()
    if edited is not None:
        for col in editable_cols:
            if col in edited.columns:
                working_df[col] = edited[col].values

    if save_validation:
        validated_rows = working_df[working_df["Validated"] == True]  # noqa: E712

        if validated_rows.empty:
            sac.alert(
                label="No rows are marked as validated. Please validate at least one row before saving.",
                icon="exclamation-triangle",
                color="warning",
            )
        else:
            user = get_current_user()
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
