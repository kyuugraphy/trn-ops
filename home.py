import streamlit as st

from utils.styles import inject_custom_css

inject_custom_css()

st.title("TrnClassification Support")
st.markdown("""
Welcome to the Transaction Classification Support Tool.

Please select an option from the sidebar to continue:
- **Manual Accounts**: Search, create, and edit manual account data entries.
- **Transaction Labeling**: Review and validate classified transactions.
""")

with st.sidebar:
    st.markdown("## TrnClassification")
    st.caption("v2.0.0 · Demo Mode")
