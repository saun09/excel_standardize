import streamlit as st
import pandas as pd
import re
from io import BytesIO
import unicodedata
from io import StringIO
from difflib import SequenceMatcher
from collections import defaultdict

# Strict email regex to avoid false positives
email_pattern = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

def is_email(value):
    value = str(value).strip()
    return bool(email_pattern.match(value))

def detect_string_columns(df):
    string_cols = []
    for col in df.columns:
        series = df[col].dropna()
        # Filter strings
        string_values = series[series.apply(lambda x: isinstance(x, str))]
        if not string_values.empty:
            # Exclude if any looks like email
            if not string_values.map(is_email).any():
                string_cols.append(col)
        # Check if column has any string with alphabetic char
        has_text = series.astype(str).apply(lambda x: any(c.isalpha() for c in x)).any()
        # Exclude columns that contain emails
        contains_email = series.astype(str).map(is_email).any()
        # Exclude numeric-only columns
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if has_text and not contains_email and not is_numeric:
            string_cols.append(col)
    return string_cols

def convert_df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def clean_pin(value):
    if pd.isna(value):
        return value
    value = str(value)
    # Remove "pin-" prefix, case-insensitive
    value = re.sub(r'pin-', '', value, flags=re.IGNORECASE).strip()
    # Extract first group of 6 digits
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
    val_str = val_str.lower()
    val_str = val_str.strip()
    val_str = re.sub(r'\s+', ' ', val_str)

    return val_str

def standardize_dataframe(df, string_cols):
    df = df.copy()
    for col in string_cols:
        df[col] = df[col].apply(lambda x: standardize_value(x, col_name=col))
    return df

# NEW CLUSTERING FUNCTIONS
def extract_core_product_name(text):
    """Extract the core product name by removing technical details and codes"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Remove content in parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove technical specifications and codes
    # Remove patterns like PQ0015066, FOR FOOTWEAR INDUSTRY, etc.
    text = re.sub(r'\b[A-Z]{2}\d+\b', '', text)  # Remove codes like PQ0015066
    text = re.sub(r'\bFOR\s+[A-Z\s]+INDUSTRY\b', '', text, flags=re.IGNORECASE)
    
    # Remove aqueous dispersion descriptions
    text = re.sub(r'\(AQUEOUS DISPERSION.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'AQUEOUS DISPERSION.*', '', text, flags=re.IGNORECASE)
    
    # Remove container information
    text = re.sub(r'\b\d+\s*(FLEXI\s*TANK|CONT|CONTAINER).*', '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    
    # Extract main product pattern (brand + model)
    # For LIPOLAN products, keep the main identifier
    match = re.match(r'([a-z]+\s*[a-z]*)\s*([a-z0-9\-]+)', text)
    if match:
        brand = match.group(1).strip()
        model = match.group(2).strip()
        return f"{brand} {model}"
    
    return text

def similarity_score(str1, str2):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, str1, str2).ratio()

def cluster_product_names(series, similarity_threshold=0.7):
    """Cluster similar product names together"""
    if series.empty:
        return pd.Series([], dtype=str)
    
    # Get unique values and their core names
    unique_values = series.dropna().unique()
    core_names = {val: extract_core_product_name(val) for val in unique_values}
    
    # Group by core names first
    core_groups = defaultdict(list)
    for val, core in core_names.items():
        if core:  # Only process non-empty core names
            core_groups[core].append(val)
    
    # Create clusters within each core group
    clusters = {}
    cluster_id = 0
    
    for core_name, values in core_groups.items():
        if len(values) == 1:
            # Single item, use core name as cluster
            clusters[values[0]] = core_name
        else:
            # Multiple items, check for sub-clustering
            processed = set()
            for val in values:
                if val in processed:
                    continue
                
                cluster_members = [val]
                processed.add(val)
                
                # Find similar items
                for other_val in values:
                    if other_val != val and other_val not in processed:
                        if similarity_score(val, other_val) >= similarity_threshold:
                            cluster_members.append(other_val)
                            processed.add(other_val)
                
                # Create cluster name (use the core name or the shortest representative)
                cluster_name = core_name if core_name else min(cluster_members, key=len)
                
                for member in cluster_members:
                    clusters[member] = cluster_name
    
    # Map the series values to cluster names
    return series.map(lambda x: clusters.get(x, extract_core_product_name(x) if pd.notna(x) else x))

def add_cluster_column(df, column_name):
    """Add a cluster column for the specified column"""
    if column_name not in df.columns:
        return df
    
    df_copy = df.copy()
    cluster_col_name = f"{column_name}_cluster"
    df_copy[cluster_col_name] = cluster_product_names(df_copy[column_name])
    
    return df_copy

# Streamlit UI starts here
st.title("Automatic String Column Standardizer with Clustering")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read CSV with fallback encoding
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    st.subheader("Original Data Sample")
    st.dataframe(df.head(10))

    string_cols = detect_string_columns(df)
    st.write(f"**Detected string columns (to standardize):** {string_cols}")

    if st.button("Standardize String Columns"):
        df_clean = standardize_dataframe(df, string_cols)
        st.subheader("Standardized Data Sample")
        st.dataframe(df_clean.head(10))

        # Prepare CSV for download
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Standardized CSV",
            data=csv,
            file_name="standardized_output.csv",
            mime="text/csv"
        )
        
        # Store standardized data in session state for clustering
        st.session_state['df_standardized'] = df_clean
        st.session_state['string_cols'] = string_cols

# Clustering section (only show if standardized data exists)
if 'df_standardized' in st.session_state:
    st.subheader("Product Name Clustering")
    st.write("Select a column to cluster similar product names:")
    
    df_std = st.session_state['df_standardized']
    cluster_column = st.selectbox(
        "Choose column for clustering:",
        options=st.session_state['string_cols'],
        key="cluster_column_select"
    )
    
    if st.button("Create Clusters"):
        df_clustered = add_cluster_column(df_std, cluster_column)
        
        st.subheader("Data with Clusters")
        # Show original, standardized, and clustered columns side by side
        display_cols = [cluster_column, f"{cluster_column}_cluster"]
        if cluster_column in df_clustered.columns:
            st.dataframe(df_clustered[display_cols].head(20))
        
        # Show cluster summary
        cluster_col = f"{cluster_column}_cluster"
        if cluster_col in df_clustered.columns:
            cluster_counts = df_clustered[cluster_col].value_counts()
            st.subheader("Cluster Summary")
            st.write(f"Total unique clusters: {len(cluster_counts)}")
            st.dataframe(cluster_counts.head(10).to_frame("Count"))
        
        # Download clustered data
        csv_clustered = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data with Clusters",
            data=csv_clustered,
            file_name="clustered_output.csv",
            mime="text/csv"
        )

st.subheader("How Clustering Works")
st.write("""
**Clustering Process:**
1. **Extract Core Names**: Removes technical details, codes in parentheses, and industry specifications
2. **Group Similar Items**: Uses text similarity to group related products
3. **Create Cluster Names**: Generates clean, searchable cluster names

**Examples:**
- `LIPOLAN F -2530 F (PQ0015066)( FOR FOOTWEAR INDUSTRY)` → `lipolan f -2530`
- `LIPOLAN F -2630 F (PQ0015066)( FOR FOOTWEAR INDUSTRY)` → `lipolan f -2630`
- `LIPOLAN T 24H70(AQUEOUS DISPERSION...)` → `lipolan t 24h70`

This allows you to search for "lipolan" products and get all variants grouped together.
""")