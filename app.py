import streamlit as st
import pandas as pd
import re
import unicodedata
from collections import Counter

# Sentence Transformers
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Session state initialization
if "standardized" not in st.session_state:
    st.session_state.standardized = False
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

def is_email(value):
    value = str(value).strip()
    return bool(email_pattern.match(value))

def detect_string_columns(df):
    string_cols = []
    for col in df.columns:
        series = df[col].dropna()
        has_text = series.astype(str).apply(lambda x: any(c.isalpha() for c in x)).any()
        contains_email = series.astype(str).map(is_email).any()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if has_text and not contains_email and not is_numeric:
            string_cols.append(col)
    return string_cols

def clean_pin(value):
    if pd.isna(value):
        return value
    value = str(value)
    value = re.sub(r'pin-', '', value, flags=re.IGNORECASE).strip()
    match = re.search(r'\b(\d{6})\b', value)
    return match.group(1) if match else value

def standardize_value(val, col_name=""):
    if pd.isna(val):
        return val
    val_str = str(val)
    if val_str.strip() == "":
        return val_str
    if "pin" in col_name.lower():
        return clean_pin(val)
    val_str = unicodedata.normalize('NFKD', val_str).encode('ascii', 'ignore').decode('utf-8')
    val_str = val_str.lower().strip()
    val_str = re.sub(r'\s+', ' ', val_str)
    return val_str

def standardize_dataframe(df, string_cols):
    df = df.copy()
    for col in string_cols:
        df[col] = df[col].apply(lambda x: standardize_value(x, col_name=col))
    return df

def cluster_with_embeddings(series, threshold=1.5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(series.fillna("").astype(str).tolist())
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
    labels = clustering.fit_predict(embeddings)
    return labels

# --- Streamlit App ---
st.title("Automatic String Column Standardizer + Smart Semantic Clustering")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.subheader("Original Data Sample")
    st.dataframe(df.head(10))

    string_cols = detect_string_columns(df)
    st.write(f"Detected string columns to standardize ({len(string_cols)}):")
    st.write(string_cols)

    if st.button("Standardize String Columns"):
        df_clean = standardize_dataframe(df, string_cols)
        st.session_state.df_clean = df_clean
        st.session_state.standardized = True

if st.session_state.standardized and st.session_state.df_clean is not None:
    df_clean = st.session_state.df_clean

    st.subheader("Standardized Data Sample")
    st.dataframe(df_clean.head(10))

    string_cols = detect_string_columns(df_clean)
    selected_col = st.selectbox("Select Column for Smart Clustering", options=string_cols)

    if st.button("Smart Cluster with Sentence Embeddings"):
        with st.spinner("Clustering with sentence embeddings..."):
            df_clean['auto_cluster'] = cluster_with_embeddings(df_clean[selected_col])

            st.subheader("Cluster Summary")
            st.dataframe(df_clean['auto_cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'auto_cluster': 'Count'}))

            st.subheader("Data Sample with Cluster Labels")
            st.dataframe(df_clean[[selected_col, 'auto_cluster']].head(20))

            csv = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV with Clusters",
                data=csv,
                file_name="semantic_clustered.csv",
                mime="text/csv"
            )
