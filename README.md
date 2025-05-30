
# 🧠 Smart Clustering & Analytics App

A **Streamlit-based data analysis tool** that allows you to **upload CSV files**, **standardize** i.e. preprocess it, perform **clustering** on selected columns, visualize the results, export color-coded Excel files, and **analyze clusters** through interactive dashboards.

---

## 📌 What Does It Do?

This Streamlit web application allows users to:
-Upload and clean raw Excel/CSV data
-Standardize messy string columns (like names / product descriptions / company names / PIN code etc.)
-Perform clustering based on any chosen column
-Generate color-coded Excel files with clusters grouped visually
-Conduct various types of cluster-based data analytics
-Group data across categorical columns for insights



---

## ⚙️ How Does It Work?

-Step 1: Upload your CSV file (click on the browse files and upload button)
-Step 2: Click on the **Standardize String Columns** button to preprocess the data(i.e. lowercase/pin code formatting/ space removal, etc.)
-Step 3: You can download the **Standardized CSV** file now
-Step 4: Choose a column for clustering your information over (i.e. item_description ) , click on the **create clusters** button.
-Step 5: You can download the Data with Clusters as CSV Or click the **Generate Color-Coded Excel** Button where each cluster is color coded and grouped together for better visualization
-Step 6: For data analysis, select analysis type [Cluster Summary/Top Clusters/Cross Analysis/Complete Analysis by category] 
    i) then select the numeric column for calculations [ i.e. quantity/ unit_price etc.]
    ii) next, select specific clusters (i.e. lipolan t , lipolan f-2530 etc) - can select multiple OR leave blank for all analysis -> Download Analysis Results
-Step 7: Data grouping: for categorical columns- Select columns to group by(multiple) 
    i) then select columns to aggregate over(like count/sum etc. - OPTIONAL)
    ii) next, click on **group data** button to get table.

---

##  What Output Will You Get?

- **Excel Output**:
  - Sheet 1: Standardized CSV   
  - Sheet 2: Clustered data CSV
  - Sheet 3: Color-coded by cluster.
 
- **Interactive Outputs**:
  - Downloadable Insights from Analysis:
    Choose your Analysis Type:
    **Cluster Summary**: Total records, sums, averages
    **Top Clusters**: Clusters with highest metrics
    **Cross Analysis**: Compare two cluster variables
    **Complete Analysis by Category**: Full view based on selected input
  - Group-by stats with aggregation-This allows you to summarize and analyze data by categories with custom aggregations.

---

## 🔁 Working Flowchart

```mermaid
flowchart TD
    A[Start: Upload CSV] --> B[Select Column for Clustering]
    B --> C[Set Clustering Params (KMeans, k)]
    C --> D[Run Clustering]
    D --> E[Preview Clustered Data]
    E --> F[Visualize Cluster Stats]
    F --> G[Generate Excel Report]
    G --> H[Advanced Cluster Analytics]
    H --> I[Group-by Aggregation]
    I --> J[Download Final Output]
```

---

## 🧾 Code Explanation

### 🔹 app.py
- **is_email(value)** : checks if a given value is a valid email address using strict regex pattern.
    It trims whitespace and converts the input to a string.

 ```python
email_pattern = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

def is_email(value):
    value = str(value).strip()
    return bool(email_pattern.match(value))



---


