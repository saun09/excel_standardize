import pandas as pd
import re
# Replace with your actual CSV file path
file_path = r'\\wsl.localhost\Ubuntu\home\saun\excel_standardize\raw.csv'

# Read CSV with fallback encoding
df = pd.read_csv(file_path, encoding='ISO-8859-1')

email_pattern = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

def is_email(value):
    value = str(value).strip()
    return bool(email_pattern.match(value))


def is_purely_numeric_column(series):
    """Check if all non-null values in the column are numeric."""
    try:
        pd.to_numeric(series.dropna())
        return True
    except ValueError:
        return False

def print_columns_with_text_values(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    print("String-like columns (excluding emails and purely numeric):")
    for col in df.columns:
        series = df[col].dropna()

        # Skip if purely numeric
        if is_purely_numeric_column(series):
            continue

        # Check if any value contains email
        if series.astype(str).map(is_email).any():
            continue

        # At least one value must contain non-digit characters
        has_text = series.astype(str).apply(lambda x: any(c.isalpha() for c in x)).any()

        if has_text:
            print(col)

print_columns_with_text_values(file_path)


import pandas as pd
import re

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

    sample_values = series.dropna().astype(str).sample(min(100, len(series)))
    for delim in delimiters:
        count = sum(val.count(delim) for val in sample_values)
        delimiter_scores[delim] = count

    best_delim = max(delimiter_scores, key=delimiter_scores.get)
    if delimiter_scores[best_delim] == 0:
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

def auto_standardize(df, string_columns, output_path=None):
    df = df.copy()

    for col in string_columns:
        print(f"Standardizing column: {col}")

        # Standardize strings
        df[col] = df[col].apply(standardize_string)

        # Extract digits for PIN/postal code columns
        if re.search(r'pin|postal|zip|code', col, re.I):
            print(f"  Extracting digits only for column {col}")
            df[col] = df[col].apply(extract_digits)

        # Split composite columns if applicable
        split_df = split_composite_column(df[col])

        if split_df is not None:
            print(f"  Splitting composite column '{col}' into {split_df.shape[1]} columns")
            split_df = split_df.rename(columns=lambda x: f"{col}_{x}")
            df = df.drop(columns=[col])
            df = pd.concat([df, split_df], axis=1)
        else:
            print(f"  Standardized {col}")

    print("\nFinal columns after standardization:")
    print(df.columns.tolist())

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nStandardized data saved to: {output_path}")

    return df
