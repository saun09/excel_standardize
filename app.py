import streamlit as st
import pandas as pd
import re
import unicodedata
import io
import random

# ---------- Helper Functions ----------

def generate_colors(n):
    random.seed(42)
    return ["#" + ''.join(random.choices('0123456789ABCDEF', k=6)) for _ in range(n)]

def detect_string_columns(df):
    string_cols = []
    for col in df.columns:
        series = df[col].dropna()
        if (series.astype(str).apply(lambda x: any(c.isalpha() for c in x)).any() 
            and not pd.api.types.is_numeric_dtype(df[col])):
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
    val_str = str(val).strip()
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

# ---------- App Title and Upload ----------

st.title("String Standardization + Exact Match Clustering")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.subheader("Original Data Sample")
    st.dataframe(df.head(10))

    string_cols = detect_string_columns(df)
    st.write(f"Detected string columns to standardize: {string_cols}")

    if st.button("Standardize String Columns"):
        st.session_state.df_clean = standardize_dataframe(df, string_cols)
        st.session_state.standardized = True

# ---------- Exact Match Clustering + Normalized Search ----------

import string
def normalize_for_clustering(val):
    if pd.isna(val):
        return "MISSING"
    val = str(val).lower()
    val = re.sub(r'[^\w\s]', '', val)  # remove punctuation
    val = re.sub(r'\s+', ' ', val)     # normalize whitespace
    val = val.replace(" ", "")         # remove all spaces for tighter matching
    return val.strip()

def normalize_cluster_name(name):
    if pd.isna(name):
        return "UNKNOWN"
    name = str(name).lower().strip()

    # Remove batch numbers (like b210119101 etc.)
    name = re.sub(r'\bb\d{9}\b', '', name)

    # Remove extra punctuation and normalize spaces
    name = name.translate(str.maketrans('', '', string.punctuation))
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

if st.session_state.get("standardized") and st.session_state.get("df_clean") is not None:
    df_clean = st.session_state.df_clean

    st.subheader("Standardized Data Sample")
    st.dataframe(df_clean.head(10))

    string_cols = detect_string_columns(df_clean)
    selected_col = st.selectbox("Select column for clustering", string_cols)

    if st.button("Exact Match Clustering"):
        df_clean['exact_cluster'] = df_clean[selected_col].apply(normalize_for_clustering)

        # Add normalized cluster column
        df_clean['normalized_cluster'] = df_clean['exact_cluster'].apply(normalize_cluster_name)

        # Create mapping normalized_cluster -> list of exact_cluster(s)
        cluster_map = df_clean.groupby('normalized_cluster')['exact_cluster'].agg(lambda x: list(set(x))).to_dict()
        normalized_options = sorted(cluster_map.keys())

        st.subheader("Exact Match Cluster Summary")
        st.dataframe(df_clean['exact_cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'exact_cluster': 'Count'}))

        unique_clusters = sorted(df_clean['exact_cluster'].unique())
        color_map = dict(zip(unique_clusters, generate_colors(min(len(unique_clusters), 20))))

        def highlight_exact_clusters(row):
            color = color_map.get(row['exact_cluster'], '#FFFFFF')
            return ['background-color: {}'.format(color) if col in [selected_col, 'exact_cluster'] else '' for col in row.index]

        styled_df = df_clean.style.apply(highlight_exact_clusters, axis=1)

        towrite = io.BytesIO()
        styled_df.to_excel(towrite, engine='openpyxl', index=False)
        towrite.seek(0)

        st.subheader("🔍 Search Data by Cluster Name (Normalized)")

        selected_normalized = st.selectbox(
            "Select a normalized cluster name to view matching rows:",
            options=normalized_options,
            index=0,
            help="Start typing to search. Similar names are grouped together."
        )

        matching_exact_names = cluster_map.get(selected_normalized, [])
        filtered_df = df_clean[df_clean['exact_cluster'].isin(matching_exact_names)]

        st.write(f"Showing {len(filtered_df)} rows for normalized cluster group:")
        st.dataframe(filtered_df)

        st.download_button(
            "Download Exact Match Clusters Excel (Colored)", 
            towrite, 
            file_name="exact_match_clusters.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
