import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(page_title="Project Breathless", layout="wide", page_icon="🌍")

# --- Helper Functions ---
@st.cache_data
def load_data():
    path = "data/processed/master_dataset.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None

@st.cache_data
def load_metrics():
    path = "results/model_evaluation_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# --- UI ---
st.title("🌍 Project Breathless: Free-Flow Visual Tool")
st.markdown("""
**Explore the data your way.** Filter by region, year, or country, and build custom visualizations to uncover hidden trends between air pollution, wealth, and health.
""")

df = load_data()

if df is None:
    st.error("Data not found. Please ensure 'data/processed/master_dataset.csv' exists.")
    st.stop()

# Sidebar Controls
st.sidebar.header("Filters & Settings")

# Year Range Slider
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Region Filter
regions = sorted(df["Region"].unique().astype(str))
selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)

# Filter Data
filtered_df = df[
    (df["Year"] >= selected_years[0]) & 
    (df["Year"] <= selected_years[1]) & 
    (df["Region"].isin(selected_regions))
]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Custom Explorer", "🗺️ Global Maps", "📊 Regional Deep Dive", "🤖 Model Performance"])

with tab1:
    st.header("Build Your Own Visuals")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-Axis", df.select_dtypes(include=np.number).columns, index=list(df.select_dtypes(include=np.number).columns).index("PM25") if "PM25" in df.columns else 0)
    with col2:
        y_axis = st.selectbox("Y-Axis", df.select_dtypes(include=np.number).columns, index=list(df.select_dtypes(include=np.number).columns).index("DeathRate") if "DeathRate" in df.columns else 0)
    with col3:
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Box", "Histogram"])

    col4, col5 = st.columns(2)
    with col4:
        color_by = st.selectbox("Color By", [None, "Region", "Country", "Year"])
    with col5:
        size_by = st.selectbox("Size By (Scatter only)", [None] + list(df.select_dtypes(include=np.number).columns))

    if chart_type == "Scatter":
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by, size=size_by, hover_name="Country", title=f"{y_axis} vs {x_axis}")
    elif chart_type == "Line":
        # Aggregate if too many points for line
        if color_by == "Country" and len(filtered_df["Country"].unique()) > 20:
            st.warning("Too many countries selected for a readable line chart. Showing top 10 by average Y-value.")
            top_countries = filtered_df.groupby("Country")[y_axis].mean().nlargest(10).index
            plot_df = filtered_df[filtered_df["Country"].isin(top_countries)]
        else:
            plot_df = filtered_df
        fig = px.line(plot_df, x=x_axis, y=y_axis, color=color_by, hover_name="Country", title=f"{y_axis} over {x_axis}")
    elif chart_type == "Box":
        fig = px.box(filtered_df, x=x_axis, y=y_axis, color=color_by, title=f"Distribution of {y_axis} by {x_axis}")
    elif chart_type == "Histogram":
        fig = px.histogram(filtered_df, x=x_axis, color=color_by, title=f"Distribution of {x_axis}")

    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Data Table"):
        st.dataframe(filtered_df)

with tab2:
    st.header("Interactive Global Maps")
    map_year = st.slider("Select Year for Map", min_year, max_year, 2019)
    map_metric = st.selectbox("Select Metric", ["PM25", "DeathRate", "GDP_per_capita", "LifeExpectancy"])
    
    map_df = df[df["Year"] == map_year]
    
    fig_map = px.choropleth(map_df, locations="Country", locationmode="country names",
                            color=map_metric,
                            hover_name="Country",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title=f"Global {map_metric} ({map_year})")
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.header("Regional Insights")
    
    # Regional Trends
    region_trends = filtered_df.groupby(["Year", "Region"])[["PM25", "DeathRate"]].mean().reset_index()
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("Avg PM2.5 by Region")
        fig_r1 = px.line(region_trends, x="Year", y="PM25", color="Region")
        st.plotly_chart(fig_r1, use_container_width=True)
    with col_r2:
        st.subheader("Avg Death Rate by Region")
        fig_r2 = px.line(region_trends, x="Year", y="DeathRate", color="Region")
        st.plotly_chart(fig_r2, use_container_width=True)

with tab4:
    st.header("Model Evaluation")
    metrics_df = load_metrics()
    
    if metrics_df is not None:
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=["R2"]), use_container_width=True)
        
        # Visualize R2 comparison
        fig_model = px.bar(metrics_df, x="Model", y="R2", title="Model R2 Score Comparison", color="Model")
        st.plotly_chart(fig_model, use_container_width=True)
    else:
        st.info("Model metrics not found. Please run the analysis notebook/script first.")
    
    st.subheader("Residual Analysis")
    if os.path.exists("results/final_figures/5_residual_plots.png"):
        st.image("results/final_figures/5_residual_plots.png", caption="Residuals from Spark ML Models")
    else:
        st.warning("Residual plot image not found.")

st.sidebar.markdown("---")
st.sidebar.info("Data processed via PySpark. Dashboard built with Streamlit.")
