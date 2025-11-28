# Project Breathless: Particulate Pythons Rules

## Project Overview
**Team:** Particulate Pythons
**Members:** Harshvardhan Rathi, Vinish Kandadi, Tyler, Colin Miller, Ryan Stefanic, Bobby
**Course:** DS/CMPSC 410 (Fall 2025)

## Data Structure
- **EPA Data:** Located in `epa/` organized by year (e.g., `epa/2010`, `epa/201four`). Contains zip files.
- **UN Data:** Located in `un data/` with various subfolders for different indicators.
- **WHO Data:**
  - Mortality rates: `who attribate deaths per 1000 standarised/data.csv`
  - PM2.5: `who pm2.5/dataall.csv`

## Environment
- **Spark:** Run in local mode with 16 cores (`local[16]`).
- **Memory:** Driver memory set to `16g`.
- **Shuffle Partitions:** Set to `128`.

## Deliverables
1. Cleaned Datasets (Parquet format)
2. PySpark Notebook (`analysis.ipynb`)
3. Regression Results
4. Visualization Outputs
5. Final Report

## Coding Standards
- Use `pyspark.sql` for data manipulation.
- Use `pyspark.ml` for machine learning.
- Ensure all dataframes are cached if reused multiple times.
- Document all transformations.
