# TrnClassification UI

Streamlit application for transaction classification support. Provides two main screens:

- **Manual Accounts**
Goal: Allow users to manually create/edit account records in the **MANUAL_ACC_DATA_CHANGES** table, which feeds into ACC_DATA_TAB_PIM.

Workflow:

User fills in a form with validated input widgets (IBAN, party type, category, purpose, etc.).
User can Query existing records by IBAN / ICO_NUM / RC_NUM.
A dataframe displays saved records; selecting a row loads its data back into the form for editing.
User can Save validated form data into the table (enriched with uni_pt_data before write).

- **Transaction Labeling**
Goal: Let users review random transactions from **TRN_CLASSIFIED_12M**, validate/correct their purpose classification, and save the labeling results into **TRN_VALIDATION**.

Workflow:

User sets filter criteria (payment type, source/dest identifiers, date range, row count, etc.).
Query fetches random transactions matching criteria, joined with a labeling table to evaluate each transaction's last validation date.
Dataframe displays results with additional editable columns:
Validated — boolean tickbox (default false)
CORRECTED_PURPOSE_SUBCAT — dropdown (purpose categories + "not_determinable")

Validate All button marks every row as validated.
Save Validation writes back: ACC_TRN_KEY, VALIDATION_TIME_STAMP, USER, and PURPOSE_SUBCAT (coalesce of corrected vs. original).
Special "uncertain" logic: A boolean widget controls whether the user sees transactions with uncertain PURPOSE_SUBCAT. Retention logic: if the last validation was "uncertain", it remains visible; once changed away from uncertain, it's hidden.


## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Databricks Delta mode

The app automatically uses Databricks SQL when these settings are present
(environment variables or `.streamlit/secrets.toml`):

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_HTTP_PATH`
- Optional: `DATABRICKS_CATALOG` (default `trn_catalog`)
- Optional: `DATABRICKS_SCHEMA` (default `trn_schema`)

Tables used in DB mode:

- `MANUAL_ACC_DATA_CHANGES`
- `ACC_DATA_TAB_PIM`
- `TRN_CLASSIFIED_12M`
- `TRN_VALIDATION`

NOTE - What to pay attention to
1) Streamlit's re-run model — every interaction re-runs the entire script. This causes "disappearing data" bugs that are very confusing at first. This app has heavy form ↔ dataframe state sync, which is the hardest pattern in Streamlit.

2) 

WHAT NEXT:
1) DB connection
2) Save:
- Validate All button marks dataframe
- Save: extract edited rows, coalesce logic
- Write to dev table with timestamp + user
- Verify written data manually
- Switch to real table
