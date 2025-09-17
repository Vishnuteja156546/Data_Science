import streamlit as st
import pandas as pd
import numpy as np
from eda import basic_summary, numeric_description, plot_hist, plot_corr_heatmap, missing_bar, download_link
from modeling import simple_automl, is_classification
from llm import call_grok
from utils import load_env, get_groq_key, get_groq_url

load_env()

st.set_page_config(page_title="AutoML Copilot", layout="wide")
st.title("AutoML Copilot — AI Data Science Assistant")
st.markdown("Upload CSV, run EDA, get LLM insights, and quick AutoML model.")

# Sidebar
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
use_grok = st.sidebar.checkbox("Enable Grok LLM insights", value=False)

if use_grok:
    grok_key = get_groq_key()
    grok_url = get_groq_url()
    if not grok_key or not grok_url:
        st.sidebar.warning("GROQ_API_KEY/GROQ_API_URL not found in .env.")
    else:
        st.sidebar.success("Grok API key loaded!")

st.sidebar.markdown("---")
st.sidebar.info("Recommended: small CSVs (<50MB).")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head(10))

    if st.button("Run EDA & Visualizations"):
        summary = basic_summary(df)
        st.metric("Rows", summary['rows'])
        st.metric("Columns", summary['columns'])
        st.write("Top missing columns")
        st.table(pd.Series(summary['top_missing']).rename("Missing Count").to_frame())
        st.write("Numeric summary")
        st.dataframe(numeric_description(df).round(3))
        st.plotly_chart(plot_corr_heatmap(df), use_container_width=True)
        st.plotly_chart(missing_bar(df), use_container_width=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            c = st.selectbox("Select numeric column for histogram", numeric_cols)
            st.plotly_chart(plot_hist(df, c), use_container_width=True)
        st.markdown(download_link(df), unsafe_allow_html=True)

    st.subheader("LLM Insights (Grok)")
    if use_grok and get_groq_key() and get_groq_url():
        if st.button("Generate LLM Insights"):
            with st.spinner("Calling Grok..."):
                head = df.head(10).to_csv(index=False)
                shape = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
                missing_summary = df.isnull().sum().sort_values(ascending=False).head(10).to_dict()
                prompt = f"""
You are a helpful data scientist assistant. Summarize the dataset and provide:
1) 3-5 high level observations,
2) Top 5 columns likely important for prediction,
3) Suggested preprocessing steps,
4) Quick modelling suggestions (algorithms + why),
5) Possible business insights to look for.

Dataset info:
{shape}

Top missing columns: {missing_summary}

Sample rows:
{head}
"""
                try:
                    resp = call_grok(prompt)
                    st.write(resp)
                except Exception as e:
                    st.error(f"Grok API call failed: {e}")
    else:
        st.info("Enable Grok in sidebar and set API key.")

    st.subheader("Quick AutoML")
    if st.button("Run Quick AutoML"):
        columns = df.columns.tolist()
        target = st.selectbox("Select target column", columns, index=len(columns)-1)
        with st.spinner("Training quick model..."):
            try:
                result = simple_automl(df, target, mode="fast")
                st.success("Model training done.")
                st.write("Problem type:", result.get("problem"))
                if result.get("problem") == "classification":
                    st.metric("Accuracy", f"{result['accuracy']:.3f}")
                    st.metric("F1 (weighted)", f"{result['f1_weighted']:.3f}")
                else:
                    st.metric("RMSE", f"{result['rmse']:.3f}")
            except Exception as e:
                st.error(f"AutoML failed: {e}")
else:
    st.info("Upload a CSV to get started.")
