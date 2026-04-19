import streamlit as st


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* ---- Global ---- */
        /* Let Streamlit's native theme (configured in config.toml) handle most colors and backgrounds */

        /* ---- Card containers ---- */
        /* We rely on st.container(border=True) and native expanders, but we can subtly style expanders */
        div[data-testid="stExpander"] {
            border: 1px solid rgba(49, 51, 63, .1);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,.04);
            background-color: #ffffff;
        }

        /* ---- Metric cards ---- */
        div[data-testid="stMetric"] {
            border: 1px solid rgba(49, 51, 63, .1);
            border-radius: 8px;
            padding: 12px 16px;
            background-color: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,.04);
        }
        div[data-testid="stMetric"] label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #6b778c;
        }

        /* ---- Section headers ---- */
        .section-header {
            font-size: 1.05rem;
            font-weight: 600;
            color: #172b4d;
            margin: 1.2rem 0 0.8rem 0;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid rgba(49, 51, 63, .1);
            padding-bottom: 4px;
        }

        /* ---- Page title ---- */
        .page-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #172b4d;
            margin-bottom: 0.2rem;
        }
        .page-subtitle {
            font-size: 0.95rem;
            color: #6b778c;
            margin-bottom: 1.5rem;
        }

        /* ---- Reduce main block padding for wider tables ---- */
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }

        /* ---- Let data editor scroll horizontally without squashing columns ---- */
        div[data-testid="stDataFrame"] > div {
            overflow-x: auto;
        }
        div[data-testid="stDataFrame"] table {
            min-width: max-content;
        }

        /* ---- Hide Streamlit branding ---- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True
        )


def section_header(label: str):
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)
