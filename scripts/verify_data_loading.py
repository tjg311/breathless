import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, regexp_replace

spark = SparkSession.builder.appName("Debug_Data_Loading").master("local[1]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def load_un_data_debug(path, label):
    print(f"--- Loading {label} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        pdf = pd.read_csv(path, header=1)
        pdf = pdf.drop(columns=["Footnotes", "Source"], errors='ignore')
        if 'Value' in pdf.columns:
            pdf['Value'] = pdf['Value'].astype(str)
        
        print(f"Pandas Columns: {pdf.columns.tolist()}")
        print(f"First row raw: {pdf.iloc[0].tolist()}")
        
        sdf = spark.createDataFrame(pdf)
        
        target_country_col = None
        if "Unnamed: 1" in sdf.columns:
            target_country_col = "Unnamed: 1"
        elif "Region/Country/Area" in sdf.columns:
            sample_val = str(pdf.iloc[0, 0]) if not pdf.empty else ""
            if sample_val.isdigit():
                 if len(sdf.columns) > 1:
                     target_country_col = sdf.columns[1]
            else:
                 target_country_col = "Region/Country/Area"
        else:
            target_country_col = sdf.columns[1] if len(sdf.columns) > 1 else sdf.columns[0]
            
        print(f"Selected Country Column: {target_country_col}")
        
        clean_sdf = sdf.select(
            trim(col(target_country_col)).alias("Country"),
            col("Year").cast("int"),
            regexp_replace(col("Value"), ",", "").cast("double").alias("Value")
        )
        
        clean_sdf.show(5, False)
        return clean_sdf

    except Exception as e:
        print(f"Error: {e}")

load_un_data_debug("un data/gdp and gdp per cap/SYB67_230_202411_GDP and GDP Per Capita.csv", "GDP")
load_un_data_debug("un data/pop surface area density/SYB67_1_202411_Population, Surface Area and Density.csv", "Population")
