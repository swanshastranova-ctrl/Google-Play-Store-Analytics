# Google Play Store Analytics Dashboard

An interactive Business Intelligence (BI) dashboard developed using **Python, Streamlit, Pandas, NumPy, Plotly and SciPy** to analyze Google Play Store application data. The dashboard provides an executive overview, interactive filters, automated insights, six visualization modules and downloadable PDF reporting.

The application automatically loads its default dataset from the repository, so users do not need to upload a CSV file manually to get started.

## Project Highlights

- **Executive Overview** – Displays key performance indicators (KPIs), category distribution, Free vs Paid app distribution and rating analysis.
- **Automatic Dataset Loading** – Loads the default Google Play Store dataset directly from the `data/` directory.
- **Optional CSV Upload** – Allows users to upload a different app dataset or an optional user reviews CSV.
- **Interactive Filters** – Filter data by app category, app type, minimum rating and minimum installs.
- **Automated Insights** – Generates descriptive observations based on the currently filtered dataset.
- **Executive PDF Report** – Generates a downloadable PDF report containing KPIs, insights and a top-category summary.
- **Six Visualization Modules**:

  1. **Clustered Heatmap** – Compares category-level performance metrics using normalized values and hierarchical clustering.
  2. **Hierarchical Sunburst** – Explores the hierarchy of Country, Category, App Type and Rating Band using the separate country-app master dataset.
  3. **Free vs Paid Radar** – Compares Free and Paid applications using percentile-normalized metrics.
  4. **Monthly Calendar Heatmap** – Visualizes the distribution of listed app update dates by month.
  5. **Hexbin Density** – Explores the relationship between application installs and reviews.
  6. **Monthly Streamgraph** – Visualizes monthly distributions of listed app update dates across categories.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard development |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Plotly | Interactive data visualizations |
| SciPy | Hierarchical clustering and statistical operations |
| Scikit-learn | Machine learning utilities |
| Matplotlib | Supporting visualization operations |
| ReportLab | PDF report generation |

## Repository Structure

```text
Google-Play-Store-Analytics/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── googleplaystore.csv
    ├── googleplaystore_user_reviews.csv
    └── task2_validated_country_app_master.csv
```

### Dataset Description

| Dataset | Description |
|---|---|
| `googleplaystore.csv` | Main Google Play Store application dataset loaded automatically by the dashboard |
| `googleplaystore_user_reviews.csv` | Optional user reviews dataset |
| `task2_validated_country_app_master.csv` | Separate country-app master dataset used by the Hierarchical Sunburst module |

**Important:** The main application dataset and Task 2 country-app master dataset must remain inside the `data/` directory with their exact filenames.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/swanshastranova-ctrl/Google-Play-Store-Analytics.git
```

Navigate to the project directory:

```bash
cd Google-Play-Store-Analytics
```

**GitHub Repository:**  
https://github.com/swanshastranova-ctrl/Google-Play-Store-Analytics

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

**Command Prompt:**

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will start locally and display a URL, usually:

```text
http://localhost:8501
```

Open this URL in your browser to access the dashboard.

## Data Loading and Configuration

The application uses automatic dataset loading.

### Default Dataset Loading

The main dataset is loaded from:

```text
data/googleplaystore.csv
```

When the application starts:

- It automatically checks whether the default dataset exists.
- If available, it loads and preprocesses the dataset.
- No manual CSV upload is required.
- The sidebar displays the active data source.

### Optional CSV Upload

The sidebar also provides optional upload controls:

- Upload a different app dataset to replace the default dataset for the current session.
- Upload an optional user reviews CSV.

If the default app dataset is missing, the application displays a message asking the user to add the dataset or upload a CSV.

### Task 2 Dataset

The Hierarchical Sunburst visualization uses a separate country-app master dataset:

```text
data/task2_validated_country_app_master.csv
```

Make sure this file is available at the specified location before accessing the Task 2 visualization.

## How to Use the Dashboard

1. Launch the application using the Streamlit run command.
2. The dashboard automatically loads the default app dataset.
3. Open **Overview** to explore KPIs, category distribution, app types and automated insights.
4. Apply interactive filters to analyze specific categories, app types, rating thresholds and install thresholds.
5. Generate and download the Executive PDF Report.
6. Select any of the six visualization modules from the sidebar.
7. Explore the interactive charts and their analytical results.
8. For the Hierarchical Sunburst module, ensure that the separate Task 2 country-app master dataset is present in the `data/` directory.

### Development Preview

The sidebar includes a **Development preview (testing only)** option.

This option temporarily bypasses the configured time restrictions for visualization modules during testing.

Keep this option disabled during normal evaluation unless testing outside the configured visualization time windows is required.

## Methodology and Interpretation Notes

- The dashboard performs data cleaning and preprocessing before generating visualizations.
- Numeric fields, dates, missing values and duplicate records are handled through preprocessing operations.
- Task-specific eligibility filters are applied where required.
- The Task 1 composite score uses Z-score-normalized category metrics and configurable weights. It is a comparative analytical index, not an official Google Play ranking.
- Weighted ratings use review counts as weights where specified.
- Revenue calculated using installs multiplied by listed price represents an indicative estimate, not actual publisher revenue.
- The review-to-install ratio is a dataset-level proxy and should not be interpreted as actual user-level engagement.
- The Calendar Heatmap and Monthly Streamgraph use the dataset's **Last Updated** field. They represent distributions of the latest listed update dates, not complete historical update-event records.
- Missing update-date observations do not necessarily mean that an application was never updated.
- The Task 2 Sunburst visualization uses the separate country-app master dataset and its eligibility filters. Its results depend on the supplied dataset.

## Troubleshooting

### 1. Default Dataset Not Found

Check whether the following file exists:

```text
data/googleplaystore.csv
```

Ensure that the filename and directory structure are correct.

### 2. Task 2 Master Dataset Missing

Verify that the file exists at:

```text
data/task2_validated_country_app_master.csv
```

### 3. Missing Python Module

Activate the virtual environment and run:

```bash
pip install -r requirements.txt
```

### 4. Application Does Not Start

- Ensure that the terminal is opened in the project root directory.
- Confirm that `app.py` and `requirements.txt` are present.
- Check the terminal for Python errors.
- Run the application again:

```bash
python -m streamlit run app.py
```

### 5. Port Already in Use

Run Streamlit on another port:

```bash
python -m streamlit run app.py --server.port 8502
```

## Future Improvements

- Add additional dataset validation and automated testing.
- Introduce more advanced statistical and predictive analytics.
- Add configurable date-based and category-based comparisons.
- Improve PDF report customization and chart export options.
- Expand the dashboard with additional country-level and application-level insights.
- Enhance deployment and performance optimization.

## Author

**Swansh Shree**

GitHub: https://github.com/swanshastranova-ctrl

---

**Project:** Google Play Store Analytics Dashboard

**Domain:** Data Analytics and Business Intelligence

**Framework:** Streamlit

**Language:** Python

---

*If you use or extend this project, please retain appropriate attribution and follow the usage terms of the datasets you distribute.*