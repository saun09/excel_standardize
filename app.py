import streamlit as st
import pandas as pd
import re
from dateutil import parser

# Helper: Clean string fields
def clean_string(value):
    value = str(value).lower().strip()
    value = re.sub(r'[\s]+', ' ', value)  # Collapse spaces
    return value

# Helper: Clean PIN-like fields
def clean_pin(value):
    if pd.isna(value):
        return value
    value = str(value)
    value = value.replace('PIN-', '').strip()
    match = re.search(r'\b(\d{6})\b', value)
    return match.group(1) if match else value

# Helper: Clean date fields
def clean_date(value):
    try:
        return parser.parse(str(value), dayfirst=True).date().isoformat()
    except Exception:
        return value

# Load file with correct encoding
@st.cache_data
def load_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(('xlsx', 'xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    return df

# Streamlit UI
st.title("Auto Data Cleaner for Excel/CSV Files")

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['xlsx', 'xls', 'csv'])

if uploaded_file is not None:
    df = load_file(uploaded_file)

    st.subheader("Preview Before Cleaning")
    st.dataframe(df.head())

    for col in df.columns:
        if df[col].dtype == object:
            # Skip emails
            if df[col].astype(str).str.contains('@').any():
                continue

            sample_vals = df[col].dropna().astype(str).head(20).tolist()

            # Handle PIN-like fields
            if any("PIN" in val or re.match(r'\d{6}', val) for val in sample_vals):
                df[col] = df[col].apply(clean_pin)

            # Handle likely dates
            elif any(re.search(r'\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4}', val) for val in sample_vals):
                df[col] = df[col].apply(clean_date)

            # Generic string cleaning
            else:
                df[col] = df[col].apply(clean_string)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())

    # Option to download cleaned file
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")

else:
    st.info("Please upload an Excel or CSV file.")
