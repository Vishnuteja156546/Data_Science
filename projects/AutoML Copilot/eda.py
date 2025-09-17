import pandas as pd
import numpy as np
import plotly.express as px
import base64

pd.options.mode.chained_assignment = None

def basic_summary(df):
    summary = {}
    summary['rows'] = df.shape[0]
    summary['columns'] = df.shape[1]
    summary['dtypes'] = df.dtypes.astype(str).to_dict()
    summary['missing'] = df.isnull().sum().to_dict()
    summary['duplicated_rows'] = int(df.duplicated().sum())
    summary['top_missing'] = df.isnull().sum().sort_values(ascending=False).head(5).to_dict()
    return summary

def numeric_description(df):
    return df.describe(include=[np.number]).T

def plot_hist(df, col):
    fig = px.histogram(df, x=col, nbins=40, title=f'Histogram: {col}')
    return fig

def plot_box(df, col):
    fig = px.box(df, y=col, points="outliers", title=f'Boxplot: {col}')
    return fig

def plot_corr_heatmap(df):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        fig = px.imshow([[0]], text_auto=True, title="Correlation: Not enough numeric columns")
        return fig
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix", zmin=-1, zmax=1)
    return fig

def missing_bar(df):
    miss = df.isnull().sum()
    miss = miss[miss>0].sort_values(ascending=False)
    if miss.empty:
        fig = px.bar(x=[0], y=[0], title="No missing values")
        return fig
    fig = px.bar(x=miss.index.astype(str), y=miss.values, title="Missing Values by Column")
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def download_link(df, filename="processed.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href
