import streamlit as st

st.set_page_config(
    page_title="TrnClassification",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Navigation": [
        st.Page("home.py", title="Home", icon="🏠"),
        st.Page("pages/1_Manual_Accounts.py", title="Manual Accounts"),
        st.Page("pages/2_Transaction_Labeling.py", title="Transaction Labeling"),
    ]
}

pg = st.navigation(pages)
pg.run()
