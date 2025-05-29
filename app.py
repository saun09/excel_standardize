import streamlit as st
import pandas as pd
import re
import unicodedata
from io import StringIO

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
        # Check if column has any string with alphabetic char
        has_text = series.astype(str).apply(lambda x: any(c.isalpha() for c in x)).any()
        # Exclude columns that contain emails
        contains_email = series.astype(str).map(is_email).any()
        # Exclude numeric-only columns
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if has_text and not contains_email and not is_numeric:
            string_cols.append(col)
    return string_cols

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

   # digits = re.sub(r'\D', '', val_str)
    #digit_ratio = len(digits) / max(len(val_str), 1)
    #if digit_ratio > 0.8:
       # return digits

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

st.title("Automatic String Column Standardizer")

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
        st.subheader("Standardized Data Sample")
        st.dataframe(df_clean.head(10))

        # Prepare CSV for download
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV",
            data=csv,
            file_name="standardized_output.csv",
            mime="text/csv"
        )
