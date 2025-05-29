import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import re
import io

st.set_page_config(page_title="Auto Excel Standardizer", layout="wide")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def standardize_all_columns(df, threshold=90):
    mappings_per_column = {}
    df_std = df.copy()
    
    for col in df.columns:
        if df[col].dtype == object:
            df_std[col + '_original'] = df[col]
            df_std[col] = df[col].apply(clean_text)
            unique_vals = df_std[col].dropna().unique()
            mapping = {}

            for val in unique_vals:
                if not mapping:
                    # First entry becomes its own representative
                    mapping[val] = val
                else:
                    result = process.extractOne(val, mapping.keys(), scorer=fuzz.token_sort_ratio)
                    if result is not None:
                        match, score, _ = result
                        if score >= threshold:
                            mapping[val] = mapping[match]
                        else:
                            mapping[val] = val
                    else:
                        mapping[val] = val

            df_std[col] = df_std[col].map(mapping)
            mappings_per_column[col] = mapping

    return df_std, mappings_per_column


# === Streamlit UI ===
st.title("🤖 Auto Excel Standardizer")

uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File loaded and read!")

    with st.expander("📊 Original Preview"):
        st.dataframe(df.head(20))

    st.info("Standardizing all string columns... please wait.")
    standardized_df, mappings = standardize_all_columns(df, threshold=90)
    st.success("🎉 Standardization complete!")

    with st.expander("🧼 Standardized Preview"):
        st.dataframe(standardized_df.head(20))

    # File download
    output = io.BytesIO()
    standardized_df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    st.download_button(
        label="📥 Download Standardized Excel",
        data=output,
        file_name="standardized_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Search Interface
    st.subheader("🔍 Search Across Standardized Data")
    search_term = st.text_input("Enter search term (e.g., 'sachin optics')")

    if search_term:
        cleaned_search = clean_text(search_term)
        result_rows = pd.DataFrame()

        for col in standardized_df.select_dtypes(include='object').columns:
            matches = standardized_df[standardized_df[col].apply(
                lambda x: fuzz.token_sort_ratio(cleaned_search, x) >= 90
            )]
            result_rows = pd.concat([result_rows, matches], ignore_index=True)

        if not result_rows.empty:
            st.success(f"🔎 Found {len(result_rows)} matching rows:")
            st.dataframe(result_rows.drop_duplicates())
        else:
            st.warning("No matches found.")
