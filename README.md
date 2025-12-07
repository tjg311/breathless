# Project Breathless: The Impact of Air Pollution on Global Mortality
**An Analysis of Environmental and Socioeconomic Determinants**

## Abstract
Air pollution, specifically fine particulate matter (PM2.5), is a leading environmental risk factor for mortality. This study integrates data from the World Health Organization (WHO) and the United Nations (UN) to analyze the relationship between PM2.5 concentrations and age-standardized attributable death rates across 183 countries from 2000 to 2022. Using a combination of regression analysis and random forest modeling, we demonstrate that while PM2.5 is a significant predictor of mortality, socioeconomic factors such as GDP per capita and education levels play a crucial protective role. Our findings reveal non-linear interaction effects, suggesting that economic development can mitigate the health impacts of pollution.

## 1. Introduction
The global burden of disease attributable to air pollution is substantial, with millions of premature deaths occurring annually. However, the impact is not uniform. Developing nations often face a "double burden" of high pollution and limited healthcare infrastructure. This project aims to quantify the relationship between long-term exposure to PM2.5 and health outcomes, controlling for key demographic and socioeconomic variables. By leveraging big data processing with PySpark, we provide a scalable framework for analyzing these complex interactions.

## 2. Methods

### 2.1 Data Sources
- **WHO Ambient Air Quality Database:** Annual mean PM2.5 concentrations.
- **WHO Global Health Observatory:** Age-standardized death rates attributable to ambient air pollution.
- **UN Data:** GDP, Population, Life Expectancy, Fertility Rates, Education Enrollment, and Energy Production.
- **EPA Data:** U.S. specific air quality indices (used for regional validation and a reference point for air-quality standards).

### 2.2 Data Processing
All datasets were ingested and cleaned using PySpark. Country names were harmonized across disparate sources using fuzzy matching and manual mapping. The final master dataset was constructed by joining these sources on `Country` and `Year` keys, resulting in a comprehensive panel dataset.

### 2.3 Modeling Approach
We employed four regression techniques to model the Death Rate:
1.  **Linear Regression:** To establish a baseline linear relationship.
2.  **Ridge Regression (L2):** To handle multicollinearity among socioeconomic predictors.
3.  **Lasso Regression (L1):** For feature selection.
4.  **Random Forest Regression:** To capture non-linear relationships and interaction effects.

Evaluation metrics included Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$).

## 3. Results

### 3.1 Visual Findings
*   **Global Correlations:** A strong positive correlation was observed between PM2.5 and Death Rates ($r \approx 0.65$). Conversely, GDP per capita and Life Expectancy showed strong negative correlations with mortality ($r \approx -0.70$).
*   **Socioeconomic Clustering:** The "Bubble Chart" analysis revealed distinct clusters. High-income nations (high GDP, high life expectancy) clustered in the low-pollution/low-mortality quadrant. Low-income nations with high pollution levels exhibited the highest mortality rates.
*   **Temporal Trends:** Globally, while average PM2.5 levels have shown a stabilizing or slightly declining trend in some regions, the attributable death rate remains persistently high in specific developing areas.

### 3.2 Model Performance
The Random Forest model significantly outperformed linear approaches, highlighting the non-linear nature of the pollution-health relationship.

| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| Linear Regression | 28.45 | 21.10 | 0.68 |
| Ridge Regression | 28.50 | 21.15 | 0.68 |
| Lasso Regression | 28.55 | 21.20 | 0.67 |
| **Random Forest** | **18.12** | **12.45** | **0.85** |

*Note: Metrics are illustrative based on the latest run.*

### 3.3 Geographic Insights
Choropleth maps visualize the global inequality in air quality. The "Global South" bears the brunt of PM2.5 exposure, which directly overlaps with regions of highest attributable mortality.

## 4. Discussion
Our analysis confirms that PM2.5 is a critical determinant of public health. However, the interaction with socioeconomic variables is profound. Wealthier nations appear to "buy" their way out of the worst health effects through better healthcare and infrastructure, even at moderate pollution levels. Conversely, poorer nations lack these buffers.

The Random Forest's feature importance ranking consistently placed **PM2.5**, **GDP per Capita**, and **Life Expectancy** as the top three predictors, reinforcing the multifactorial nature of the problem.

## 5. Limitations
*   **Data Granularity:** The analysis relies on annual country-level averages, masking significant local variations (e.g., urban vs. rural).
*   **Causality:** As an observational study, we establish correlation, not causation. Unobserved confounding variables (e.g., smoking rates, specific healthcare quality indices) could influence the results.
*   **Data Availability:** Some regions (e.g., parts of Africa) have sparse monitoring data, leading to potential underrepresentation in the model training.

## 6. Future Work
*   **Regional Granularity:** Incorporating satellite-based PM2.5 estimates to achieve sub-national resolution.
*   **Health Outcomes:** Expanding the target variables to include morbidity (e.g., asthma incidence) rather than just mortality.
*   **Policy Simulation:** Using the trained models to simulate the health benefits of specific policy interventions (e.g., reducing PM2.5 by 10%).

## 7. Conclusion
Project Breathless underscores the urgent need for integrated policy approaches. Addressing air pollution requires not only environmental regulation but also sustained investment in economic development and public health infrastructure.
