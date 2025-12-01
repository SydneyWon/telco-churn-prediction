# Telco Customer Churn Prediction


🎯 Project Overview
This project focuses on developing, comparing, and optimizing Machine Learning classification models to proactively identify customers at high risk of churning in a competitive telecommunications environment.

The core goal was to shift the modeling objective from maximizing simple accuracy to minimizing the costly business error of False Negatives (missed churners), thereby maximizing the potential Return on Investment (ROI) from targeted customer retention efforts.


📈 Key Achievements & Insights
* Strategic Metric Focus: Adopted Recall (Sensitivity) and AUC-ROC as the primary evaluation metrics, acknowledging that the financial cost of a False Negative (losing a customer's lifetime value) is significantly higher than that of a False Positive (wasting a marketing budget).
  
* Feature Engineering for Business Insight: Created new features derived from the data, such as:
  * Monthly_to_Total_Ratio: Captured the high churn risk associated with the initial customer lifecycle phase (low tenure).
  * Has_No_Security: Identified a high-risk segment with low service stickiness, exhibiting a churn rate near $50\\%$.

* Optimal Model Selection: The final model, Optimized Logistic Regression, was selected based on its superior balance between predictive performance (highest Recall of $0.799$ and AUC-ROC of $0.849$) and high interpretability, which is crucial for operationalizing business decisions.

* Actionable Strategy: Translated model outputs into immediate, high-impact business recommendations, focusing on contract conversion and enhancing service stickiness for high-risk customers.


🛠️ Technology Stack & Libraries
* Language: Python
* Core Libraries:
  * pandas, numpy: Data Manipulation and Numerical Operations
  * matplotlib, seaborn: Data Visualization and EDA
  * scikit-learn: Preprocessing Pipelines (ColumnTransformer, StandardScaler, OneHotEncoder), Model Selection, and Evaluation.
  * xgboost, lightgbm: High-Performance Gradient Boosting Models


📊 Data Source
The project utilizes the well-known Telco Customer Churn Dataset.
* Source: Kaggle: Telco Customer Churn
* Key Challenge: The dataset features a significant class imbalance ($73.46\\%$ Non-Churn vs. $26.54\\%$ Churn). This was addressed through Stratified Sampling and Class Weighting during model tuning.


⚙️ Methodology & Modeling
1. Data Cleaning & Preprocessing:
* Handled missing values in TotalCharges using verified business logic (imputing 0.0 for customers with 0 tenure).
* Categorized features and built a robust ColumnTransformer for standardization and one-hot encoding.
* Split data using StratifiedKFold to maintain class balance in all cross-validation folds.
  
2. Feature Engineering: Created the high-impact features Monthly_to_Total_Ratio and Has_No_Security to capture lifecycle risk and service stickiness.
   
3. Model Training & Optimization:
* Trained and compared four distinct models: Logistic Regression, Random Forest, XGBoost, and LightGBM.
* Utilized class_weight='balanced' and scale_pos_weight in tuning to prioritize Recall.
* Employed GridSearchCV and RandomizedSearchCV to fine-tune hyperparameters, optimizing for the AUC-ROC metric.
