import streamlit as st
import pandas as pd
import re
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

# Initialize model once (expensive)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Normalize function
def normalize_name(name):
    name = str(name).lower()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    suffixes = ['incorporated', 'inc', 'corp', 'corporation', 'co', 'company', 'limited', 'ltd', 'llc']
    for suf in suffixes:
        name = re.sub(r'\b' + suf + r'\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Canonical name selector
def get_canonical_name(names):
    return min(names, key=len)

# Streamlit UI
st.title("Company Name Standardizer & Query Tool")

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['xlsx', 'xls', 'csv'])

    
if uploaded_file is not None:
    # Load file depending on type
    if uploaded_file.name.endswith(('xls', 'xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            except Exception as e:
                st.error(f"Failed to read CSV file: {e}")
                st.stop()
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Check if required columns exist
    required_columns = ['Company', 'Product', 'Year']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Normalize company names
        df['Company_norm'] = df['Company'].apply(normalize_name)

        # Generate embeddings
        embeddings = model.encode(df['Company_norm'].tolist())

        # Cluster company names
        clustering = DBSCAN(eps=0.4, min_samples=1, metric='cosine').fit(embeddings)
        df['cluster'] = clustering.labels_

        # Map canonical names
        canonical_map = {}
        for cluster_id in df['cluster'].unique():
            cluster_names = df.loc[df['cluster'] == cluster_id, 'Company_norm']
            canonical = get_canonical_name(cluster_names)
            for name in cluster_names:
                canonical_map[name] = canonical
        df['Company_standardized'] = df['Company_norm'].map(canonical_map)

        st.write("### Standardized Company Names (Sample)")
        st.dataframe(df[['Company', 'Company_standardized']].drop_duplicates().reset_index(drop=True))

        # Query input
        st.sidebar.header("Search Query")
        query_company = st.sidebar.text_input("Company Name (standardized)", "")
        query_product = st.sidebar.text_input("Product", "")
        year_range = st.sidebar.slider("Year range", int(df['Year'].min()), int(df['Year'].max()),
                                      (int(df['Year'].min()), int(df['Year'].max())))

        if st.sidebar.button("Run Query"):
            query_df = df.copy()

            if query_company:
                # Normalize query_company same way + map to canonical
                norm_query = normalize_name(query_company)
                std_query = canonical_map.get(norm_query, norm_query)
                query_df = query_df[query_df['Company_standardized'] == std_query]

            if query_product:
                query_df = query_df[query_df['Product'].str.lower() == query_product.lower()]

            query_df = query_df[query_df['Year'].between(year_range[0], year_range[1])]

            st.write(f"### Query Results: {len(query_df)} rows")
            st.dataframe(query_df)

else:
    st.info("Please upload an Excel (.xlsx/.xls) or CSV (.csv) file with columns: Company, Product, Year")
