from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Alphabetagama AI Group Finder", layout="wide"
)

# Application theme: white background with red text.
# Streamlit's default theme can vary by machine, so the selectors use
# !important to keep the POC appearance consistent.
st.markdown(
    """
    <style>
        /* White page background */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background-color: #ffffff !important;
        }

        /* Red text throughout the application */
        html, body, p, label, div, span, small,
        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"] {
            color: #c00000 !important;
        }

        /* Text typed into the interest box */
        textarea, input {
            color: #c00000 !important;
            background-color: #ffffff !important;
            caret-color: #c00000 !important;
        }

        /* Red border for the interest box */
        textarea:focus, input:focus {
            border-color: #c00000 !important;
            box-shadow: 0 0 0 1px #c00000 !important;
        }

        /* Keep buttons clear and consistent */
        button {
            color: #c00000 !important;
            background-color: #ffffff !important;
            border: 1px solid #c00000 !important;
        }

        /* Ensure table text is also red */
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] * {
            color: #c00000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
DATA_FILE = Path(__file__).parent / "dataset.xlsx"


@st.cache_data
def load_data(file_bytes):
    df = pd.read_excel(BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    required_columns = [
        "Group Display Name", "Is Deleted",
        "Group Type", "Owner Principal Name",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError("Missing Excel columns: " + ", ".join(missing))
    df["POC ID"] = [
        f"group-{i:04d}" for i in range(1, len(df) + 1)
    ]
    deleted = df["Is Deleted"].astype(str).str.strip().str.upper()
    group_type = df["Group Type"].astype(str).str.strip().str.lower()
    df = df.loc[
        deleted.eq("FALSE") & group_type.eq("public")
    ].copy()
    df["Group Display Name"] = (
        df["Group Display Name"].fillna("").astype(str).str.strip()
    )
    df = df.loc[df["Group Display Name"].ne("")].copy()
    for column in ["Group Description", "Topics"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    df["Search Text"] = (
        df["Group Display Name"] + ". "
        + df["Group Description"] + ". " + df["Topics"]
    )
    return df.reset_index(drop=True)


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_data
def build_embeddings(texts):
    model = load_model()
    return model.encode(
        list(texts), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )


def find_groups(interest, df):
    model = load_model()
    group_embeddings = build_embeddings(tuple(df["Search Text"]))
    interest_embedding = model.encode(
        interest, normalize_embeddings=True, convert_to_numpy=True
    )
    scores = group_embeddings @ interest_embedding
    positions = np.argsort(scores)[::-1][:5]
    return [
        {
            "id": df.iloc[position]["POC ID"],
            "score": float(scores[position]),
        }
        for position in positions
    ]


def display_value(column, value):
    if pd.isna(value):
        return "Not available"
    if "Date" in column or "date" in column:
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
        if not pd.isna(parsed):
            return parsed.strftime("%d/%m/%Y")
    if column in [
        "Member Count", "Posted Count", "Read Count",
        "Liked Count", "Report Period",
    ]:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            pass
    return str(value)


if not DATA_FILE.exists():
    st.error("dataset.xlsx was not found beside app.py.")
    st.stop()
try:
    df = load_data(DATA_FILE.read_bytes())
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()
if df.empty:
    st.warning("No Public, non-deleted groups are available.")
    st.stop()

if "results" not in st.session_state:
    st.session_state.results = []
if "selected_group" not in st.session_state:
    st.session_state.selected_group = None


if st.session_state.selected_group is not None:
    selected = df.loc[
        df["POC ID"].eq(st.session_state.selected_group)
    ]
    if selected.empty:
        st.warning("This group is no longer available.")
    else:
        group = selected.iloc[0]
        st.title(group["Group Display Name"])
        st.caption("Synthetic Viva Engage group details")
        if group["Group Description"]:
            st.write(group["Group Description"])
        if group["Topics"]:
            st.write("Topics:", group["Topics"])
        excluded_columns = {
            "POC ID", "Search Text", "Group Display Name",
            "Group Description", "Topics",
        }
        details = [
            {"Field": c, "Value": display_value(c, group[c])}
            for c in df.columns if c not in excluded_columns
        ]
        st.dataframe(pd.DataFrame(details), hide_index=True)
    if st.button("Back to recommendations"):
        st.session_state.selected_group = None
        st.rerun()
    st.stop()


st.title("Alphabetagama AI Group Finder")
st.write("Enter your interests to discover relevant groups.")
st.caption(f"Searching {len(df)} Public, non-deleted groups.")
with st.form("search_form"):
    interest = st.text_area(
        "What are you interested in?",
        placeholder="Example: detecting fraud in motor claims.",
        height=120,
    )
    submitted = st.form_submit_button("Find top 5 groups")

if submitted:
    st.session_state.results = []
    if not interest.strip():
        st.warning("Please enter an interest first.")
    else:
        try:
            with st.spinner("Finding relevant groups..."):
                st.session_state.results = find_groups(
                    interest.strip(), df
                )
            st.session_state.last_interest = interest.strip()
        except Exception as error:
            st.error(
                "Unable to run the AI model. The first run "
                "requires internet access to download it."
            )
            with st.expander("Show technical error"):
                st.code(str(error))

if st.session_state.results:
    st.subheader("Your top recommendations")
    st.write("Interest:", st.session_state.last_interest)
    st.caption(
        "Results are ranked by similarity. "
        "The score is not a confidence percentage."
    )
    for rank, result in enumerate(st.session_state.results, start=1):
        matching = df.loc[df["POC ID"].eq(result["id"])]
        if matching.empty:
            continue
        group = matching.iloc[0]
        with st.container(border=True):
            st.subheader(f"{rank}. {group['Group Display Name']}")
            st.caption(f"Similarity score: {result['score']:.3f}")
            if st.button(
                "View group details", key=f"details_{result['id']}"
            ):
                st.session_state.selected_group = result["id"]
                st.rerun()
