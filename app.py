import streamlit as st
import pandas as pd
import re
from io import BytesIO

# Helper functions (same as before)
def is_email(value):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", str(value)))

def standardize_string(value):
    if pd.isna(value):
        return value
    val = str(value).strip().lower()
    return val

def extract_digits(value):
    if pd.isna(value):
        return value
    digits = re.findall(r'\d+', str(value))
    return ''.join(digits) if digits else None

def split_composite_column(series):
    delimiters = [',', ';', '/', '|', '-', '(', ')']
    delimiter_scores = {}

    sample_values = series.dropna().astype(str).sample(min(100, len(series))) if len(series.dropna()) > 0 else pd.Series(dtype=str)
    for delim in delimiters:
        count = sum(val.count(delim) for val in sample_values)
        delimiter_scores[delim] = count

    best_delim = max(delimiter_scores, key=delimiter_scores.get) if delimiter_scores else None
    if not best_delim or delimiter_scores[best_delim] == 0:
        return None

    if best_delim in ['(', ')']:
        splits = series.astype(str).str.split(r'\(')
        splits = splits.apply(lambda parts: [p.strip().rstrip(')') for p in parts])
    else:
        splits = series.astype(str).str.split(best_delim)

    max_parts = splits.apply(len).max()

    df_parts = pd.DataFrame(
        splits.tolist(),
        columns=[f'part{i+1}' for i in range(max_parts)],
        index=series.index
    ).fillna('')

    return df_parts

def auto_standardize(df, string_columns):
    df = df.copy()
    for col in string_columns:
        df[col] = df[col].apply(standardize_string)
        if re.search(r'pin|postal|zip|code', col, re.I):
            df[col] = df[col].apply(extract_digits)
        split_df = split_composite_column(df[col])
        if split_df is not None:
            split_df = split_df.rename(columns=lambda x: f"{col}_{x}")
            df = df.drop(columns=[col])
            df = pd.concat([df, split_df], axis=1)
    return df

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
    return string_cols

def convert_df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# Streamlit UI starts here
st.title("Auto Standardizer for CSV Data")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    # Read CSV with fallback encoding
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    st.write("### Original Data Sample")
    st.dataframe(df.head())

    string_cols = detect_string_columns(df)

    st.write(f"**Detected string columns (to standardize):** {string_cols}")

    if st.button("Run Standardization"):
        df_std = auto_standardize(df, string_cols)
        st.write("### Standardized Data Sample")
        st.dataframe(df_std.head())

        csv_bytes = convert_df_to_csv_bytes(df_std)
        st.download_button(
            label="Download Standardized CSV",
            data=csv_bytes,
            file_name="standardized_output.csv",
            mime="text/csv"
        )
