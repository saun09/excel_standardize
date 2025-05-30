
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


**Step 1:** Upload your CSV file (click on the **Browse Files and Upload** button)  
**Step 2:** Click on the **Standardize String Columns** button to preprocess the data (i.e., lowercase/pin code formatting/space removal, etc.)  
**Step 3:** You can download the **Standardized CSV** file now  
**Step 4:** Choose a column for clustering your information over (e.g. `item_description`), then click on the **Create Clusters** button  
**Step 5:**  
- You can download the **Data with Clusters** as a CSV  
- OR click the **Generate Color-Coded Excel** button where each cluster is color-coded and grouped together for better visualization  

**Step 6: For data analysis:**  
- Select analysis type: *Cluster Summary / Top Clusters / Cross Analysis / Complete Analysis by Category*  
  i) Select the numeric column for calculations (e.g., `quantity`, `unit_price`, etc.)  
  ii) Select specific clusters (e.g. `lipolan t`, `lipolan f-2530`, etc.) — can select multiple OR leave blank for full analysis  
- Click **Download Analysis Results**

**Step 7: Data Grouping (for categorical columns):**  
  i) Select columns to group by (multiple allowed)  
  ii) Select columns to aggregate over (like count/sum, etc. — *optional*)  
  iii) Click on the **Group Data** button to get the final grouped table  

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
    - It trims whitespace and converts the input to a string.
    - Uses a regex pattern to match typical email formats
    - Returns True if the value looks like an email else false
    - We do this to ensure that email addresses are not standardized, since they are case sensitive.
  

 ```python
email_pattern = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

def is_email(value):
    value = str(value).strip()
    return bool(email_pattern.match(value))
<!-- spacer -->


- **detect_string_columns(df)** : identifies columns in a dataframe that contain non-numeric and non email text values.
  

---


