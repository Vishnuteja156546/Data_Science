# AI Data Copilot

AI Data Copilot is a lightweight portfolio project that turns a CSV, TSV, or Excel upload into an automated data science workflow.

## Features

- CSV, TSV, XLSX, and XLS upload with session creation
- Dataset profiling, semantic column detection, preview, missing values, duplicates, outliers, and health score
- Automatic EDA charts for distributions, categories, scatter plots, time trends, and correlations
- Cleaning with duplicate removal, median/mode fills, and outlier capping
- Groq-powered dataset summary, business insights, and dataset chat when `GROQ_API_KEY` is configured
- Local fallback insights when Groq is not configured
- RandomForest AutoML for classification or regression
- Simple trend forecasting for datetime plus numeric datasets
- Processed CSV, EDA report, business report, and prediction report downloads

## Setup

1. Add your Groq key in `backend/.env`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server from the project root:

```bash
uvicorn backend.main:app --reload
```

4. Open `http://127.0.0.1:8000`.

## Environment

The app reads settings from `backend/.env`. `GROQ_API_KEY` can be empty while testing non-LLM features.
