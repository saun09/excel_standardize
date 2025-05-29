import streamlit as st
import pandas as pd
import re
from io import BytesIO
import unicodedata
from io import StringIO
from difflib import SequenceMatcher
from collections import defaultdict
import openpyxl
from openpyxl.styles import PatternFill
import random

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
    """Extract the core product name by preserving important product codes"""
    if pd.isna(text):
        return ""
    
    original_text = str(text).strip()
    text = original_text.lower()
    
    # First, extract important product codes before removing parentheses
    # Look for patterns like (ar-740), (ar-825h), (pq0015066), etc.
    product_codes = []
    
    # Extract alphanumeric codes with hyphens (like ar-740, ar-825h)
    code_matches = re.findall(r'\(([a-z]{2,3}-?\d+[a-z]*)\)', text)
    product_codes.extend(code_matches)
    
    # Extract other product codes (like pq0015066)
    other_codes = re.findall(r'\(([a-z]{2}\d+)\)', text)
    product_codes.extend(other_codes)
    
    # Remove descriptions in parentheses but keep the main text structure
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Extract the base product name (like "acm", "lipolan f", etc.)
    base_name = ""
    
    # Try to match common patterns
    if re.match(r'^[a-z]+\s*[a-z]*', text):
        # Extract first 1-2 words as base name
        words = text.split()
        if len(words) >= 2:
            base_name = f"{words[0]} {words[1]}"
        else:
            base_name = words[0] if words else ""
    else:
        base_name = text
    
    # Combine base name with the most specific product code
    if product_codes:
        # Prioritize codes with hyphens and letters (more specific)
        specific_codes = [code for code in product_codes if '-' in code and any(c.isalpha() for c in code)]
        if specific_codes:
            return f"{base_name} {specific_codes[0]}"
        else:
            return f"{base_name} {product_codes[0]}"
    
    return base_name

def similarity_score(str1, str2):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, str1, str2).ratio()

def cluster_product_names(series, similarity_threshold=0.8):
    """Cluster similar product names together with better product code handling"""
    if series.empty:
        return pd.Series([], dtype=str)
    
    # Get unique values and their core names
    unique_values = series.dropna().unique()
    core_names = {val: extract_core_product_name(val) for val in unique_values}
    
    # Create direct mapping - each unique core name becomes a cluster
    clusters = {}
    
    for val, core in core_names.items():
        if core and core.strip():  # Only process non-empty core names
            clusters[val] = core
        else:
            # Fallback for items without clear core names
            clusters[val] = str(val).lower().strip()
    
    # Map the series values to cluster names
    return series.map(lambda x: clusters.get(x, str(x).lower().strip() if pd.notna(x) else x))

def add_cluster_column(df, column_name):
    """Add a cluster column for the specified column"""
    if column_name not in df.columns:
        return df
    
    df_copy = df.copy()
    cluster_col_name = f"{column_name}_cluster"
    df_copy[cluster_col_name] = cluster_product_names(df_copy[column_name])
    
    return df_copy

def generate_colors(n):
    """Generate n distinct colors for clusters"""
    colors = [
        'FFE6E6', 'E6F3FF', 'E6FFE6', 'FFF0E6', 'F0E6FF',
        'FFFFE6', 'FFE6F0', 'E6FFFF', 'F0FFE6', 'FFE6CC',
        'E6E6FF', 'CCFFE6', 'FFE6B3', 'E6CCFF', 'B3FFE6',
        'FFB3E6', 'B3E6FF', 'E6FFB3', 'FFB3CC', 'CCFFB3',
        'FFD700', 'FFB6C1', '98FB98', 'DDA0DD', 'F0E68C',
        'FFA07A', '20B2AA', 'FFE4B5', 'D3D3D3', 'F5DEB3'
    ]
    
    if n <= len(colors):
        return colors[:n]
    else:
        # Generate additional random colors if needed
        additional_colors = []
        for _ in range(n - len(colors)):
            color = f"{random.randint(200, 255):02X}{random.randint(200, 255):02X}{random.randint(200, 255):02X}"
            additional_colors.append(color)
        return colors + additional_colors

def create_colored_excel(df, cluster_column):
    """Create an Excel file with color-coded clusters"""
    cluster_col = f"{cluster_column}_cluster"
    
    if cluster_col not in df.columns:
        return None
    
    # Sort by cluster to group similar items together
    df_sorted = df.sort_values(by=cluster_col).reset_index(drop=True)
    
    # Get unique clusters and assign colors
    unique_clusters = df_sorted[cluster_col].unique()
    colors = generate_colors(len(unique_clusters))
    cluster_colors = dict(zip(unique_clusters, colors))
    
    # Create Excel file in memory
    output = BytesIO()
    
    # Write to Excel
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write main data sheet
        df_sorted.to_excel(writer, sheet_name='Clustered_Data', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Clustered_Data']
        
        # Apply colors to rows based on clusters
        cluster_col_idx = df_sorted.columns.get_loc(cluster_col) + 1  # +1 for Excel 1-based indexing
        
        for row in range(2, len(df_sorted) + 2):  # Start from row 2 (after header)
            cluster_value = df_sorted.iloc[row-2][cluster_col]
            color_hex = cluster_colors.get(cluster_value, 'FFFFFF')
            fill = PatternFill(start_color=color_hex, end_color=color_hex, fill_type='solid')
            
            # Apply color to entire row
            for col in range(1, len(df_sorted.columns) + 1):
                worksheet.cell(row=row, column=col).fill = fill
        
        # Create a summary sheet with cluster information
        cluster_summary = df_sorted.groupby(cluster_col).size().reset_index(name='Count')
        cluster_summary['Color'] = cluster_summary[cluster_col].map(cluster_colors)
        cluster_summary.to_excel(writer, sheet_name='Cluster_Summary', index=False)
        
        # Apply colors to summary sheet
        summary_sheet = writer.sheets['Cluster_Summary']
        for row in range(2, len(cluster_summary) + 2):
            cluster_value = cluster_summary.iloc[row-2][cluster_col]
            color_hex = cluster_colors.get(cluster_value, 'FFFFFF')
            fill = PatternFill(start_color=color_hex, end_color=color_hex, fill_type='solid')
            
            for col in range(1, len(cluster_summary.columns) + 1):
                summary_sheet.cell(row=row, column=col).fill = fill
    
    output.seek(0)
    return output.getvalue()

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
        
        # Store clustered data in session state
        st.session_state['df_clustered'] = df_clustered
        st.session_state['cluster_column_name'] = cluster_column
        
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
        
        # Download clustered data as CSV
        csv_clustered = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data with Clusters (CSV)",
            data=csv_clustered,
            file_name="clustered_output.csv",
            mime="text/csv"
        )

# Show clustered data and Excel export if clustering has been done
if 'df_clustered' in st.session_state:
    df_clustered = st.session_state['df_clustered']
    cluster_column = st.session_state['cluster_column_name']
    
    # Show the clustered data again
    st.subheader("Clustered Data (Persistent)")
    display_cols = [cluster_column, f"{cluster_column}_cluster"]
    st.dataframe(df_clustered[display_cols].head(20))
    
    # Show cluster summary
    cluster_col = f"{cluster_column}_cluster"
    cluster_counts = df_clustered[cluster_col].value_counts()
    st.write(f"**Total unique clusters:** {len(cluster_counts)}")
    
    # Color-Coded Excel Export Section
    st.subheader("Color-Coded Excel Export")
    st.write("Generate an Excel file where each cluster is color-coded and grouped together:")
    
    if st.button("Generate Color-Coded Excel", key="excel_export"):
        with st.spinner("Creating color-coded Excel file..."):
            excel_data = create_colored_excel(df_clustered, cluster_column)
            
            if excel_data:
                st.session_state['excel_data'] = excel_data
                st.session_state['excel_ready'] = True
                st.success("✅ Excel file generated successfully!")
    
    # Show download button if Excel is ready
    if st.session_state.get('excel_ready', False) and 'excel_data' in st.session_state:
        st.download_button(
            label="📊 Download Color-Coded Excel File",
            data=st.session_state['excel_data'],
            file_name="clustered_data_colored.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Show preview of what the Excel will contain
        cluster_col = f"{cluster_column}_cluster"
        preview_data = df_clustered.groupby(cluster_col).size().reset_index(name='Row_Count')
        st.write("**Excel File Contents:**")
        st.write("- **Clustered_Data** sheet: All rows grouped by cluster with color coding")
        st.write("- **Cluster_Summary** sheet: Summary of clusters with counts")
        st.write("**Cluster Distribution:**")
        st.dataframe(preview_data.head(10))

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

**Color-Coded Excel Export:**
- All rows with the same cluster are grouped together
- Each cluster gets a unique color (green, yellow, blue, etc.)
- Two sheets: 'Clustered_Data' (main data) and 'Cluster_Summary' (overview)
- Easy to visually identify product groups for analysis

This allows you to search for "lipolan" products and get all variants grouped together with visual color coding!
""")