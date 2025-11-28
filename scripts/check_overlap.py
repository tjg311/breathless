import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower

spark = SparkSession.builder.appName("DebugData").master("local[1]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load WHO Mortality
who_df = spark.read.option("header", "true").option("inferSchema", "true").csv("who attribate deaths per 1000 standarised/data.csv")
who_countries = who_df.select(lower(trim(col("Location"))).alias("Country")).distinct().toPandas()['Country'].tolist()

# Load UN GDP
un_gdp_path = "un data/gdp and gdp per cap/SYB67_230_202411_GDP and GDP Per Capita.csv"
pdf = pd.read_csv(un_gdp_path, header=1)
# UN data country col is usually the second one, or named "Region/Country/Area"
col_name = "Region/Country/Area" if "Region/Country/Area" in pdf.columns else pdf.columns[1]
un_countries = pdf[col_name].astype(str).str.strip().str.lower().unique().tolist()

print(f"WHO Countries (Sample): {who_countries[:5]}")
print(f"UN Countries (Sample): {un_countries[:5]}")

# Check overlap
overlap = set(who_countries).intersection(set(un_countries))
print(f"Overlap Count: {len(overlap)}")
print(f"WHO Total: {len(who_countries)}")
print(f"UN Total: {len(un_countries)}")

if len(overlap) < 10:
    print("CRITICAL: Very low overlap. Check naming conventions.")
