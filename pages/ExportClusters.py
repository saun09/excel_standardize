import streamlit as st
import pandas as pd
import io

st.title("Export Clusters as Separate Excel Sheets")

# Check if clustering has been done and data is available
if "df_clean" not in st.session_state or st.session_state.df_clean is None:
    st.warning("Please upload data and perform clustering first on the Home page.")
elif 'nuanced_cluster' not in st.session_state.df_clean.columns:
    st.warning("No clustering found. Please run the 'Nuanced Clustering' on the Home page first.")
else:
    df_clean = st.session_state.df_clean

    st.write("Preview of clustered data:")
    st.dataframe(df_clean.head(10))

    if st.button("Export Excel: Each Cluster in Separate Sheet"):
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            for cluster_id, group_df in df_clean.groupby('nuanced_cluster'):
                sheet_name = f"Cluster_{cluster_id}"
                # Excel sheet name max length is 31
                sheet_name = sheet_name[:31]
                group_df.to_excel(writer, sheet_name=sheet_name, index=False)
            # Optionally add all data sheet
            df_clean.to_excel(writer, sheet_name='All_Data', index=False)
        towrite.seek(0)

        st.download_button(
            label="Download Excel with Clusters as Sheets",
            data=towrite,
            file_name="clusters_grouped_sheets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
