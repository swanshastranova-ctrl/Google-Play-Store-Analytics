# Google Play Store Analytics Dashboard

An interactive analytics dashboard built with **Python, Streamlit, Pandas, NumPy and Plotly** to explore Google Play Store app data through six visualizations, an executive overview, automated dataset insights, interactive filters and PDF reporting.

## Project Highlights

- **Executive Overview** – headline KPIs, category distribution, Free vs Paid mix and rating distribution.
- **Interactive filters** – filter the overview by app category, app type, minimum rating and minimum installs.
- **Automated Insights** – generate descriptive observations from the currently filtered dataset.
- **Executive PDF Report** – download a PDF containing KPIs, insights and a top-category summary.
- **Six visualization modules**:
  1. Clustered Heatmap – compare category-level metrics using normalized values and hierarchical clustering.
  2. Hierarchical Sunburst – explore country, category, app type and rating-band levels using the separate country-app master dataset.
  3. Free vs Paid Radar – compare app types using percentile-normalized metrics.
  4. Monthly Calendar Heatmap – inspect the distribution of listed app update dates by month.
  5. Hexbin Density – explore the relationship between installs and reviews.
  6. Monthly Streamgraph – explore monthly app update-date distributions across categories.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- SciPy
- ReportLab (PDF export)

## Repository Structure

```text
Google-Play-Store-Analytics/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── googleplaystore.csv
    ├── googleplaystore_user_reviews.csv
    └── task2_validated_country_app_master.csv
```

The user-reviews CSV is included as an optional supporting dataset. The current dashboard's main visualizations use the app dataset and the separate Task 2 country-app master dataset.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/swanshastranova-ctrl/Google-Play-Store-Analytics.git
cd Google-Play-Store-Analytics
```

Repository: https://github.com/swanshastranova-ctrl/Google-Play-Store-Analytics

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) that you can open in your browser.

## Data Setup

Keep the datasets in the `data/` directory using these exact filenames:

| File | Purpose |
|---|---|
| `googleplaystore.csv` | Main Google Play Store app dataset uploaded in the sidebar |
| `googleplaystore_user_reviews.csv` | Optional user-review dataset |
| `task2_validated_country_app_master.csv` | Separate country-app dataset used by the hierarchical sunburst module |

The Task 2 file must be located at:

```text
data/task2_validated_country_app_master.csv
```

If a dataset is not included in a local clone, obtain it from an authorized source and place it at the expected path before running the corresponding feature.

## How to Use

1. Start the Streamlit application.
2. Upload `googleplaystore.csv` using the sidebar file uploader.
3. Open **Overview** to inspect the KPIs, charts and automated insights.
4. Apply the overview filters to focus on a category, app type, rating threshold or install threshold.
5. Use **Download Executive PDF Report** to export a summary for the currently filtered overview data.
6. Select any of the six analysis modules from the sidebar to explore the detailed visualizations.
7. For Task 2, ensure the separate country-app master CSV is present in the `data/` folder.

> **Testing note:** The sidebar includes a Development preview option for testing task pages outside their configured time windows. Keep this option disabled during normal evaluation unless the project instructions specifically require preview mode.

## Methodology and Interpretation Notes

- The dashboard cleans numeric fields and dates before analysis and applies task-specific eligibility filters.
- The Task 1 composite score uses Z-score-normalized category metrics and user-adjustable weights. The score is a comparative index, not an official Google Play ranking.
- Weighted ratings use review counts as weights where specified.
- Revenue estimates based on installs multiplied by listed price are indicative estimates, not actual publisher revenue.
- Review-to-install ratio is a dataset-level proxy. It should not be interpreted as a user-level engagement rate.
- The calendar and streamgraph modules use the dataset's **Last Updated** field. These are distributions of latest listed update dates, not complete historical update-event logs. A missing date observation does not prove that no app update occurred.
- The Task 2 sunburst uses the separate country-app master dataset and its eligibility field/filters. Its country-level values depend on the supplied dataset.

## Troubleshooting

**Task 2 says the master dataset is missing**

Check that this exact path exists:

```text
data/task2_validated_country_app_master.csv
```

**A Python module is missing**

Activate your virtual environment and run:

```bash
pip install -r requirements.txt
```

**The app does not start**

Check the terminal for the first traceback, confirm that you are in the repository root (the folder containing `app.py`), and run:

```bash
python -m streamlit run app.py
```

## Future Improvements

- Add additional dataset validation and automated tests.
- Add configurable date and category comparisons.
- Improve report customization and chart export options.
- Add deployment configuration for Streamlit Community Cloud or another hosting platform.

## Author

**Swansh Shree**

---

If you use or extend this project, please retain appropriate attribution and check the usage terms of the datasets you distribute.
