import streamlit as st
import pandas as pd
import re
import unicodedata
from collections import Counter
import io
# Add imports for clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import random

# Helper function to generate distinct colors
def generate_colors(n):
    random.seed(42)
    colors = []
    for _ in range(n):
        colors.append("#"+"".join([random.choice('0123456789ABCDEF') for _ in range(6)]))
    return colors

def color_clusters(s, cluster_colors):
    # s is a Series of cluster labels
    return ['background-color: {}'.format(cluster_colors.get(x, '#FFFFFF')) for x in s]

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

def infer_common_tokens(series, top_k=15):
    tokens = []
    for val in series.dropna():
        val = re.sub(r'[^\w\s]', '', str(val)).lower()
        tokens.extend(val.split())
    counter = Counter(tokens)
    common_tokens = [tok for tok, count in counter.items() if count > 1]
    return sorted(common_tokens, key=counter.get, reverse=True)[:top_k]

def extract_primary_token(val):
    if pd.isna(val):
        return "MISC"
    val = str(val).lower()
    val = re.sub(r'[^\w\s]', ' ', val)
    tokens = val.split()
    skip_words = {'for', 'in', 'used', 'material', 'industry', 'binder', 'bulk', 'with', 'from'}
    for tok in tokens:
        if tok not in skip_words and len(tok) > 2 and any(c.isalpha() for c in tok):
            return tok.upper()
    return "MISC"

def tokenize_string(val):
    if pd.isna(val):
        return []
    val = val.lower()
    val = re.sub(r'[^\w\s]', ' ', val)
    tokens = val.split()
    stopwords = {'for', 'in', 'used', 'material', 'industry', 'binder', 'bulk', 'with', 'from'}
    return [t for t in tokens if t not in stopwords and len(t) > 2]
st.title("Automatic String Column Standardizer + Token-Based Clustering")

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
    selected_col = st.selectbox("Select Column for Token-Based Clustering", options=string_cols)

    # Your old token clustering button here (unchanged) ...

    if st.button("Nuanced Clustering by TF-IDF + DBSCAN"):
        with st.spinner("Performing nuanced clustering..."):
            df_clean['tokens_str'] = df_clean[selected_col].apply(lambda x: ' '.join(tokenize_string(x)))

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df_clean['tokens_str'])

            clustering_model = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
            clusters = clustering_model.fit_predict(X)

            df_clean['nuanced_cluster'] = clusters

            st.subheader("Nuanced Cluster Summary")
            st.dataframe(df_clean['nuanced_cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'nuanced_cluster': 'Count'}))

            st.subheader("Sample Data with Nuanced Cluster Labels")
            st.dataframe(df_clean[[selected_col, 'nuanced_cluster']].head(20))

            # Prepare colors for clusters
            unique_clusters = df_clean['nuanced_cluster'].unique()
            unique_clusters_sorted = sorted(unique_clusters)
            colors = generate_colors(len(unique_clusters_sorted))
            cluster_color_map = dict(zip(unique_clusters_sorted, colors))

            # Styling function to color by cluster in the selected column only
            def highlight_clusters(row):
                color = cluster_color_map.get(row['nuanced_cluster'], '#FFFFFF')
                return ['background-color: {}'.format(color) if col == selected_col or col == 'nuanced_cluster' else '' for col in row.index]

            # Apply styling
            styled_df = df_clean.style.apply(highlight_clusters, axis=1)

            # Export styled excel to bytes buffer
            towrite = io.BytesIO()
            styled_df.to_excel(towrite, engine='openpyxl', index=False)
            towrite.seek(0)

            st.download_button(
                label="Download Nuanced Clusters Excel (Colored)",
                data=towrite,
                file_name="nuanced_clustered_colored.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        if st.button("Exact Match Clustering (Group identical values)"):
            with st.spinner("Performing exact match clustering..."):
        # Assign the cluster ID as the group label by exact string value in selected_col
                df_clean['exact_cluster'] = df_clean[selected_col].fillna("MISSING").astype(str)

                st.subheader("Exact Match Cluster Summary")
                st.dataframe(df_clean['exact_cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'exact_cluster': 'Count'}))

                st.subheader("Sample Data with Exact Match Cluster Labels")
                st.dataframe(df_clean[[selected_col, 'exact_cluster']].head(20))

        # Generate colors for distinct clusters (limited to 20 distinct colors for performance)
                unique_clusters = df_clean['exact_cluster'].unique()
                unique_clusters_sorted = sorted(unique_clusters)
                colors = generate_colors(min(len(unique_clusters_sorted), 20))
                cluster_color_map = dict(zip(unique_clusters_sorted, colors))

                def highlight_exact_clusters(row):
                    color = cluster_color_map.get(row['exact_cluster'], '#FFFFFF')
                    return ['background-color: {}'.format(color) if col == selected_col or col == 'exact_cluster' else '' for col in row.index]

                styled_df = df_clean.style.apply(highlight_exact_clusters, axis=1)

                towrite = io.BytesIO()
                styled_df.to_excel(towrite, engine='openpyxl', index=False)
                towrite.seek(0)

                st.download_button(
                label="Download Exact Match Clusters Excel (Colored)",
                data=towrite,
                file_name="exact_match_clusters_colored.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
