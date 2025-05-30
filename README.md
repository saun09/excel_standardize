
# Smart Clustering & Analytics 

A **Streamlit-based data analysis tool** that allows you to **upload CSV files**, **standardize** i.e. preprocess it, perform **clustering** on selected columns, visualize the results, export color-coded Excel files, and **analyze clusters** through interactive dashboards.

---
## 📑 Index

1. [What Does It Do?](#what-does-it-do)
2. [How Does It Work?](#how-does-it-work)
3. [What Output Will You Get?](#what-output-will-you-get)
4. [Working Flowchart](#working-flowchart)
5. [Clustering Approach](#clustering-approach)
   - [How Clustering Was Done](#how-clustering-was-done)
   - [Why Not KMeans or DBSCAN?](#why-not-kmeans-or-dbscan)
   - [Example](#example)
6. [Code Explanation](#code-explanation)
---
## What Does It Do?

This Streamlit web application allows users to:
-Upload and clean raw Excel/CSV data
-Standardize messy string columns (like names / product descriptions / company names / PIN code etc.)
-Perform clustering based on any chosen column
-Generate color-coded Excel files with clusters grouped visually
-Conduct various types of cluster-based data analytics
-Group data across categorical columns for insights

---
## How Does It Work?

**Step 1:** Upload your CSV file (click on the **Browse Files and Upload** button)  
**Step 2:** Click on the **Standardize String Columns** button to preprocess the data (i.e., lowercase/pin code formatting/space removal, etc.)  
**Step 3:** You can download the **Standardized CSV** file now  
**Step 4:** Choose a column for clustering your information over (e.g. `item_description`), then click on the **Create Clusters** button  
**Step 5:**  
- You can download the **Data with Clusters** as a CSV  
- OR click the **Generate Color-Coded Excel** button where each cluster is color-coded and grouped together for better visualization  

**Step 6: For data analysis:**  
- Select analysis type: *Cluster Summary / Top Clusters / Cross Analysis / Complete Analysis by Category*  
  - i) Select the numeric column for calculations (e.g. `quantity`, `unit_price`, etc.)  
  - ii) Select specific clusters (e.g. `lipolan t`, `lipolan f-2530`, etc.) — can select multiple OR leave blank for full analysis  
- Click **Download Analysis Results**

**Step 7: Data Grouping (for categorical columns):**  
- i) Select columns to group by (multiple allowed)  
- ii) Select columns to aggregate over (like count/sum, etc. — *optional*)  
- iii) Click on the **Group Data** button to get the final grouped table

---

##  What Output Will You Get?

- **Excel Output**:
  - Sheet 1: Standardized CSV  
  - Sheet 2: Clustered data CSV  
  - Sheet 3: Color-coded by cluster.

- **Interactive Outputs**:
  - Downloadable Insights from Analysis:
    - **Cluster Summary**: Total records, sums, averages  
    - **Top Clusters**: Clusters with highest metrics  
    - **Cross Analysis**: Compare two cluster variables  
    - **Complete Analysis by Category**: Full view based on selected input
  - Group-by stats with aggregation - This allows you to summarize and analyze data by categories with custom aggregations.
---

##  Working Flowchart

![image](https://github.com/user-attachments/assets/af840978-44ac-4419-bae6-ad2d5a70de7d)


---
## Clustering Approach

This project performs **product name clustering** to group similar or equivalent items that may have inconsistent or messy naming conventions. The goal is to improve analysis accuracy and simplify reporting by consolidating variants of the same product under a common label.

### How Clustering Was Done

I used a **custom hybrid approach** that combines:

1. **Rule-Based Preprocessing**:
   - Extracts the **core product name** from each entry using a custom parser.
   - Preserves important identifiers like **product codes** (e.g.`AR-740`, `PQ0015066`) before removing less relevant parts (e.g. descriptions in parentheses).

2. **Fuzzy Matching via String Similarity**:
   - We use Python's `SequenceMatcher` from the `difflib` module to compute the **similarity score** between product names.
   - A **similarity threshold (default: 0.8)** is used to decide whether two product names should be grouped into the same cluster.

3. **Dictionary-Based Clustering**:
   - After extracting normalized core names, we map each original product name to a **cluster label** (core name with the most specific code).
   - This avoids over-generalization and ensures product uniqueness is preserved.

The result is a new column (e.g. `product_name_cluster`) where similar variants are grouped under a standardized label.

---

### Why Not KMeans or DBSCAN?

I chose this rule-based string similarity method instead of traditional clustering algorithms like **KMeans** or **DBSCAN** for the following reasons:

| Limitation            | KMeans / DBSCAN                              | Our Approach                                  |
|-----------------------|----------------------------------------------|-----------------------------------------------|
| **Data Type**         | Requires numeric vectors                     | Works directly with raw strings               |
| **Text Semantics**    | Cannot inherently understand product codes   | Product codes are preserved and emphasized    |
| **Feature Engineering**| Requires complex vectorization (e.g., TF-IDF) | No need for external vectorization            |
| **Clustering Control**| No control over cluster names or meaning     | Full control via rule-based mapping           |
| **Interpretability**  | Cluster labels are abstract (e.g. cluster_1)| Cluster labels are meaningful (e.g. `lipolan f AR-740`) |

In short, **textual product data** needs **semantic understanding**, not just geometric distance in vector space. My method allows for **greater flexibility, domain control, and interpretability**, which is essential for product-centric datasets.

---

### Example

| Raw Product Name                   | Cluster Label         |
|-----------------------------------|------------------------|
| `Lipolan F (AR-740)`              | `lipolan f AR-740`     |
| `Lipolan F AR740`                 | `lipolan f AR-740`     |
| `LIPOLAN F (PQ0015066)`           | `lipolan f PQ0015066`  |
| `ACM (Sample Product)`            | `acm`                  |

This clustered output is then used for:
- Summary statistics
- Cluster-wise Excel exports
- Category-wise breakdowns
---

## Code Explanation

### 🔹 app.py
- **is_email(value)** : checks if a given value is a valid email address using strict regex pattern.
  - It trims whitespace and converts the input to a string.
  - Uses a regex pattern to match typical email formats.
  - Returns True if the value looks like an email else false.
  - We do this to ensure that email addresses are not standardized, since they are case sensitive.

- **detect_string_columns(df)** : identifies columns in a dataframe that contain non-numeric and non email text values.
  - Filters out numeric columns and skips columns containing email.
  - Ensures column has atleast some alphabetic content.
  - Returns a list of column names that contains general strings.
  - We do this to get names of cholumns that have to be preprocessed.

- **detect_numeric_columns(df)** : detects columns that are either numeric like or contain numeric like strings ex: Rs1000.
  - Directly includes numeric dtype columns.
  - For non numeric columns : clean them (remove symbols) , and check if 70%+ can be converted to float.
  - Returns list of numeric like columns.
  - We do this to get definite column names to work on in the analytical insights section.

- **detect_categorical_columns(df, exclude_clusters=True)** : finds columns that are used for categorical grouping.
  - skips numeric columns.
  - check the columns unique value ratio.
  - returns list of categorical columns.
  - We do this to get column names to work on in the group by insights section.

- **safe_numeric_conversion(series)**: converts Pandas Series to numeric values ; handling formatting issues
  - replaces non numeric characters like " , , $ " and converts to float
  - NaN values become 0
  - Returns a clean numeric series

- **clean_pin(value)**: Cleans and extracts 6 digit PIN codes from strings
  - removes prefixes like "pin-"
  - finds first 6 digit number using regex
  - returns cleaned PIN or original if not found

- **standardize_value(val,col_name="")**: standardizes string values for consistency(lowercase/no accents/ clean spaces etc.)
  - if column name includes "pin" , apply clean_pin()
  - normalizes unicode characters
  - converts to lowercase and strips extra whitespace

- **standardize_dataframe(df, string_cols)**: standardizes select text columns across a dataframe
  - applies standardize_value() to each value in specified column

- **group_data(df, group_by_columns, aggregation_rules=None)** : groups a dataFrame by one or more columns with optional aggregration logic (for the group by insights section)

- **extract_core_product_name(text)** : Extracts a base product name and any product codes from a text string.
  - Extracts code using regex
  - removes parantesis content but retains product name
  - construcrs a clean product identifier
  - Real world datasets contain extra info like paranthesis. We use this functon to standardize such data for clustering.

- **similarity_score(str1, str2)** : Measures how similar 2 strings are (0 to 1)
  - Uses difflib.SequenceMatcher
  - returns a similarity score where 1 means identical strings
  - We use this for detecting near duplicates ; further helping in clustering

- **cluster_product_names(series, similarity_threshold = 0.8)** :  Clusters similar product names together by extracting a common base name and grouping variants under one label.
  - Extracts core product names using extract_core_product_name
  - maps each value in the series to its core version
  - returns series with grouped cluster names
  - We use this function to reduce noisy product name variations into standardized groups

- **add_cluster_column(df,column_name)** : Adds a new column to the dataFrame with clustered names derived from the original column.
  - calls cluster_product_names() on specified column
  - creates a new column like product_name_cluster
  - We use this to visualise the grouped identifiers for similar products

- **generate_colors(n)** : Generates a list of n distinct hex color codes to represent different product clusters for data readability and visualization

- **create_colored_excel(df, cluster_column)** : Exports clustered Dataframe to an Excel file where each row is color-coded by cluster.
  - sorts dataFrame by cluster
  - assigns unique color to each cluster
  - applies colors row-wise
  - adds summary sheet showing count and color per cluster

- **perform_cluster_analysis(df, cluster_col, analysis_type, target_col=None, group_by_col=None, selected_clusters=None)** : Analytical Insights section
  - Performs high-level summaries and breakdowns of clustered data, supporting multiple types of analysis.  
    Supported Analysis Types:
    - cluster_summary: Count, sum, and mean for each cluster.
    - top_clusters: Top 10 clusters based on a numeric field.
    - cluster_by_category: Cross-tab between clusters and a category (e.g., product type, region).
    - detailed_breakdown: Deep dive into each cluster by category with record counts and sums.
























---


