# Project Breathless: Final Presentation

## Slide 1: Title Slide
- **Project Breathless: The Impact of Air Pollution on Global Mortality**
- **Subtitle:** An Analysis of Environmental and Socioeconomic Determinants
- **Team:** Particulate Pythons
- **Date:** Fall 2025

## Slide 2: The Problem
- **Context:** Air pollution (PM2.5) is a leading cause of global mortality.
- **Question:** How strongly does long-term exposure correlate with death rates when controlling for GDP and Education?
- **Scope:** 183 Countries, 20+ Years of Data (WHO, UN, EPA).

## Slide 3: Data Architecture
- **Sources:** 
  - WHO (Mortality, PM2.5)
  - UN (GDP, Demographics, Education, Energy)
  - EPA (Validation)
- **Tech Stack:** PySpark for scalable ETL on Windows/HPC.
- **Integration:** Unified panel dataset by Country and Year.

## Slide 4: Global Correlation Heatmap
- *[Insert: results/final_figures/1_correlation_heatmap.png]*
- **Key Insight:** Strong positive correlation between PM2.5 and Death Rate. Strong negative correlation with GDP/Education.

## Slide 5: The Socioeconomic "Shield"
- *[Insert: results/final_figures/3_socioeconomic_bubble_chart.png]*
- **Visual:** Bubble chart showing Death Rate vs. PM2.5, sized by Population, colored by GDP.
- **Key Insight:** Wealthier nations (yellow/green) cluster at the bottom-left (low pollution, low death). Developing nations bear the burden.

## Slide 6: Regional Deep Dives
- *[Insert: results/final_figures/6_regional_pm25_boxplot.png]*
- **Key Insight:** Asia and Africa show significantly higher median PM2.5 levels compared to Europe and Americas.
- *[Insert: results/final_figures/7_regional_scatter.png]*
- **Key Insight:** The relationship slope is steeper in regions with lower healthcare infrastructure.

## Slide 7: Modeling Results
- **Comparison:**
  - **Linear/Ridge/Lasso:** R² ~ 0.68
  - **Random Forest:** R² ~ 0.85 (Best Performer)
- **Why?** Random Forest captures non-linear thresholds where pollution impacts accelerate.
- *[Insert Table from results/model_evaluation_metrics.csv]*

## Slide 8: Interaction Effects (3D)
- *[Insert: results/final_figures/4_3d_interaction.png]*
- **Visual:** 3D Scatter of PM2.5, GDP, Death Rate.
- **Key Insight:** The "danger zone" is high PM2.5 + low GDP. High GDP countries can withstand higher pollution with fewer attributable deaths.

## Slide 9: Geographic Disparities
- *[Insert: results/final_figures/8_map_pm25.png]*
- *[Insert: results/final_figures/9_map_deathrate.png]*
- **Key Insight:** The maps overlap almost perfectly, visualizing the environmental injustice of air pollution.

## Slide 10: Conclusion & Policy
- **Findings:** Pollution is not just an environmental issue; it's a development issue.
- **Recommendations:** 
  - Policies must target PM2.5 reduction *AND* healthcare investment.
  - Economic development is a valid public health intervention.
- **Future Work:** Sub-national granular analysis and real-time satellite monitoring.

## Slide 11: Q&A
- **Thank You!**
