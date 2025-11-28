import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visual Polish
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
palette = sns.color_palette("viridis")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_date, year, avg, lag, coalesce, when, regexp_replace, lower, trim
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def run_modeling():
    print("Initializing Spark...")
    spark = (SparkSession.builder
             .appName("Breathless_Modeling_Recovery")
             .master("local[16]")
             .config("spark.driver.memory", "16g")
             .config("spark.sql.shuffle.partitions", "128")
             .getOrCreate())

    print("Loading Data...")
    # Helper function to load UN CSVs
    def load_un_data(path, value_col_name, series_filter=None):
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            return None
        
        try:
            pdf = pd.read_csv(path, header=1)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None
        
        pdf = pdf.drop(columns=["Footnotes", "Source"], errors='ignore')
        
        if 'Value' in pdf.columns:
            pdf['Value'] = pdf['Value'].astype(str)
        else:
            return None
        
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

        clean_sdf = sdf.select(
            trim(col(target_country_col)).alias("Country"),
            col("Year").cast("int"),
            col("Series"),
            regexp_replace(col("Value"), ",", "").cast("double").alias("Value")
        )
        
        if series_filter:
            clean_sdf = clean_sdf.filter(col("Series") == series_filter)
            clean_sdf = clean_sdf.withColumnRenamed("Value", value_col_name).drop("Series")
            
        return clean_sdf

    # Load Datasets
    who_mortality_path = "who attribate deaths per 1000 standarised/data.csv"
    who_df = spark.read.option("header", "true").option("inferSchema", "true").csv(who_mortality_path)
    who_clean_df = who_df.select(
        trim(col("Location")).alias("Country"),
        col("Period").cast("int").alias("Year"),
        col("FactValueNumeric").alias("DeathRate")
    ).filter(col("Dim1") == "Both sexes")

    who_pm25_path = "who pm2.5/dataall.csv"
    who_pm25_df = spark.read.option("header", "true").option("inferSchema", "true").csv(who_pm25_path)
    who_pm25_clean = who_pm25_df.select(
        trim(col("Location")).alias("Country"),
        col("Period").cast("int").alias("Year"),
        col("FactValueNumeric").alias("PM25")
    ).filter(col("Dim1") == "Total")

    un_gdp_path = "un data/gdp and gdp per cap/SYB67_230_202411_GDP and GDP Per Capita.csv"
    un_gdp = load_un_data(un_gdp_path, "GDP", "GDP in current prices (millions of US dollars)")
    un_gdp_capita = load_un_data(un_gdp_path, "GDP_per_capita", "GDP per capita (US dollars)")

    un_pop_path = "un data/pop surface area density/SYB67_1_202411_Population, Surface Area and Density.csv"
    un_pop_raw = load_un_data(un_pop_path, "Value")
    un_pop = un_pop_raw.filter(col("Series") == "Population mid-year estimates (millions)").withColumnRenamed("Value", "Population").drop("Series")
    un_pop_density = un_pop_raw.filter(col("Series") == "Population density").withColumnRenamed("Value", "PopDensity").drop("Series")

    un_demographics_path = "un data/pop grrowth, fertility, life expectancy/SYB67_246_202411_Population Growth, Fertility and Mortality Indicators.csv"
    un_demo_raw = load_un_data(un_demographics_path, "Value")
    un_life_exp = un_demo_raw.filter(col("Series") == "Life expectancy at birth for both sexes (years)").withColumnRenamed("Value", "LifeExpectancy").drop("Series")
    un_fertility = un_demo_raw.filter(col("Series") == "Total fertility rate (children per woman)").withColumnRenamed("Value", "Fertility").drop("Series")
    
    un_edu_path = "un data/education at primary, secondary, tertiary/SYB67_309_202411_Education.csv"
    un_edu_raw = load_un_data(un_edu_path, "Value")
    un_edu_primary = un_edu_raw.filter(col("Series") == "Gross enrollment ratio - Primary (male)").withColumnRenamed("Value", "Education_Primary_Male").drop("Series") 
    
    un_energy_path = "un data/energy/SYB67_263_202411_Production, Trade and Supply of Energy.csv"
    un_energy = load_un_data(un_energy_path, "Energy_Supply", "Primary energy production (petajoules)") 

    print("Joining Data...")
    master_df = who_clean_df.join(who_pm25_clean, ["Country", "Year"], "inner")
    join_type = "left"
    datasets = [un_gdp, un_gdp_capita, un_pop, un_pop_density, un_life_exp, un_fertility, un_edu_primary, un_energy]
    
    for df in datasets:
        if df is not None:
            master_df = master_df.join(df, ["Country", "Year"], join_type)

    master_df.cache()
    count = master_df.count()
    print(f"Master DataFrame Rows: {count}")
    
    if count == 0:
        print("Error: Master DataFrame is empty. Check joins.")
        return

    print("Starting Modeling...")
    numeric_cols = [c for c, t in master_df.dtypes if t in ['int', 'double', 'float'] and c != 'Year']
    feature_cols = [c for c in numeric_cols if c != "DeathRate"]
    
    # Filter out columns that are all null
    valid_feature_cols = []
    for c in feature_cols:
        # Check if column has at least one non-null value
        # Using a quick check logic
        if master_df.select(c).dropna().count() > 0:
            valid_feature_cols.append(c)
        else:
            print(f"Dropping column {c} because it is entirely null (join mismatch).")
            
    if not valid_feature_cols:
        print("Error: No valid feature columns found after checking for nulls.")
        return

    feature_cols = valid_feature_cols
    
    # Impute missing values
    print("Imputing missing values...")
    imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols)
    # Note: Imputer in Spark 3.3+ automatically handles fitting on dataset with nulls
    try:
        model_df = imputer.fit(master_df).transform(master_df)
    except Exception as e:
        print(f"Imputation failed: {e}")
        return

    # Drop rows where DeathRate is null
    model_df = model_df.dropna(subset=["DeathRate"])
    
    final_count = model_df.count()
    print(f"Data Points for Modeling after Imputation: {final_count}")
    
    if final_count == 0:
        print("Error: No data after dropping null labels.")
        return

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    (train_data, test_data) = model_df.randomSplit([0.8, 0.2], seed=42)

    models = {
        "Linear": LinearRegression(featuresCol="scaledFeatures", labelCol="DeathRate", regParam=0.0),
        "Ridge": LinearRegression(featuresCol="scaledFeatures", labelCol="DeathRate", regParam=0.1, elasticNetParam=0.0),
        "Lasso": LinearRegression(featuresCol="scaledFeatures", labelCol="DeathRate", regParam=0.1, elasticNetParam=1.0),
        "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="DeathRate", numTrees=100)
    }

    results = []
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name}...")
        pipeline = Pipeline(stages=[assembler, model]) if name == "Random Forest" else Pipeline(stages=[assembler, scaler, model])
        
        try:
            model_fit = pipeline.fit(train_data)
            predictions = model_fit.transform(test_data)
            
            eval_rmse = RegressionEvaluator(labelCol="DeathRate", predictionCol="prediction", metricName="rmse")
            eval_mae = RegressionEvaluator(labelCol="DeathRate", predictionCol="prediction", metricName="mae")
            eval_r2 = RegressionEvaluator(labelCol="DeathRate", predictionCol="prediction", metricName="r2")
            
            rmse = eval_rmse.evaluate(predictions)
            mae = eval_mae.evaluate(predictions)
            r2 = eval_r2.evaluate(predictions)
            
            results.append((name, rmse, mae, r2))
            
            # Residual Plot
            preds_pd = predictions.select("DeathRate", "prediction").toPandas()
            preds_pd["Residuals"] = preds_pd["DeathRate"] - preds_pd["prediction"]
            sns.scatterplot(x="prediction", y="Residuals", data=preds_pd, ax=axes[i], alpha=0.5)
            axes[i].axhline(0, color='red', linestyle='--')
            axes[i].set_title(f"{name} Residuals")
            axes[i].set_xlabel("Predicted Death Rate")
            axes[i].set_ylabel("Residual (Actual - Predicted)")
            
        except Exception as e:
            print(f"Error training {name}: {e}")

    print("Saving Results...")
    if not os.path.exists("results/tables"):
        os.makedirs("results/tables")
    if not os.path.exists("results/final_figures"):
        os.makedirs("results/final_figures")

    plt.tight_layout()
    plt.savefig("results/final_figures/5_residual_plots.png", dpi=300)
    print("Saved residual plots.")

    results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
    results_df.to_csv("results/model_evaluation_metrics.csv", index=False)
    results_df.to_csv("results/tables/model_comparison.csv", index=False)
    print("Saved metrics.")
    print(results_df)

if __name__ == "__main__":
    run_modeling()
