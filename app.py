import streamlit as st
import pandas as pd
import re
import unicodedata
from io import StringIO

from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np

# Session state initialization
if "standardized" not in st.session_state:
    st.session_state.standardized = False
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# Email detection pattern
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

def standardize_company_name(name):
    name = str(name).lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def cluster_company_names(names_std):
    if len(names_std) < 2:
        return [-1] * len(names_std)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(names_std)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    clusters = clusterer.fit_predict(embeddings)
    return clusters

def get_canonical_name(cluster_names):
    return min(cluster_names, key=len)

# --- Streamlit App ---

st.title("Automatic String Column Standardizer + Company Name Clustering")

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
        company_col = st.selectbox("Select Company Name Column for Clustering", options=string_cols)

        if st.button("Cluster Company Names"):
            with st.spinner("Clustering company names..."):
                df_clean['company_std'] = df_clean[company_col].apply(standardize_company_name)
                unique_names = df_clean['company_std'].dropna().unique().tolist()
                clusters = cluster_company_names(unique_names)

                cluster_dict = {}
                for name, cluster_id in zip(unique_names, clusters):
                    if cluster_id == -1:
                        cluster_id = max(clusters) + 1 + len(cluster_dict)
                    cluster_dict.setdefault(cluster_id, []).append(name)

                canonical_mapping = {}
                for cluster_id, names_list in cluster_dict.items():
                    canonical = get_canonical_name(names_list)
                    for n in names_list:
                        canonical_mapping[n] = canonical

                df_clean['company_canonical'] = df_clean['company_std'].map(canonical_mapping)

                st.subheader("Company Name Clusters and Canonical Names")
                cluster_summary = pd.DataFrame(
                    [(cid, ', '.join(names[:5]) + ('...' if len(names) > 5 else ''), get_canonical_name(names), len(names))
                     for cid, names in cluster_dict.items()],
                    columns=['Cluster ID', 'Sample Names', 'Canonical Name', 'Count']
                )
                st.dataframe(cluster_summary)

                st.subheader("Data Sample with Canonical Company Name")
                st.dataframe(df_clean[[company_col, 'company_canonical']].head(20))

                csv = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV with Canonical Company Names",
                    data=csv,
                    file_name="standardized_and_clustered.csv",
                    mime="text/csv"
                )
