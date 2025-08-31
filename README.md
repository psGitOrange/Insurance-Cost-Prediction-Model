## Insurance-Cost-Prediction-Model
### Business Problem Statement:
Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. However, traditional methods of cost prediction often rely on broad actuarial tables and historical averages, which may not account for the nuanced differences among individuals.

Leverage machine learning techniques, to predict more accurately the insurance costs tailored to individual profiles (health conditions and fitness triats), leading to more competitive pricing and better risk management.

**Concepts Used:**

- Univariate,  Bivariate Analysis
- 2-sample t-test, Annova test, Chi-square test
- Linear Regression, Random Forest, XGBoost
- Cross Validation, Hyperparameter Tuning

[Medium Blog Post](https://medium.com/@pavansingu007/predict-health-insurance-cost-with-machine-learning-and-streamlit-ac95e0ff6b33)

### Running Streamlit App 
Run the command:
```bash
streamlit run app.py
```
Streamlit launches the app on a local port 8501, making it ready to serve predictions in real time.

**Data Profile:**
Numeric :- Age (18 to 66 years), Height (145 cm to 188 cm), Weight (51 kg to 132 kg), Premium Price (₹15,000 to ₹40,000 )

Binary (0 or 1):- Diabetes, Blood Pressure Problems, Any Transplants, Any Chronic Diseases, Known Allergies, History Of Cancer In Family, Number Of Major Surgeries

### Summary

**Key Insights from Data Analysis**

- Premium Price showed an irregular distribution with multiple peaks, suggesting **distinct premium groups**

- **Strongest** drivers of Premium Price: are Age, Transplants, Number of Major Surgeries, and Chronic Diseases
- **Weak/Very Weak relationships:** Blood Pressure Problems, Height, Diabetes, and Family History of Cancer
- **Other relationships:** Age linked with Major Surgeries and moderately with Diabetes & Blood Pressure Problems. Blood Pressure Problems & Cancer History linked with Major Surgeries
- **BMI & Premiums:** 70% policyholders were overweight/obese, at higher premium ranges

**Statistical Tests**

* **Two-Sample t-test:**
  * Premiums significantly higher for those with **Transplants (\~₹31,763)**, **Chronic Diseases (\~₹27,112)**, and **Cancer History (\~₹25,758)** compared to those without
* **ANOVA:**
  * Premiums increased significantly with **number of major surgeries** (\~₹28,000 for 2-3 surgeries)
* **Chi-Square Test:**
  * No significant association between **chronic diseases and cancer history**, or between **BMI and Diabetes/Blood Pressure Problems**

**Machine Learning Models**

* **Random Forest:**
  * Top 5 Features: **Age, Transplants, Weight, BMI, Chronic Diseases**
  * R2: **0.86** | **0.75 (CV after tuning)**

* **XGBoost:**
  * Top 5 Features: **Age, Transplants, Cancer History, Chronic Diseases, Major Surgeries**
  * R2: **0.85** | **0.73 (CV after tuning)**
* Both models showed **good predictive power** with random forest performing better. Age and Transplants consistently emerging as key factors

### Recommendations

- **Age-based Pricing:** Premiums should rise progressively with age since it is the strongest predictor. Create clear premium bands by age groups (e.g., 20–35, 36–50, 51–65)

- **High-Risk Medical History:** Apply premium loadings (additional costs)
  - Policyholders with **transplants** or **multiple surgeries** must be priced at a premium due to significantly higher costs.  
  - **Chronic diseases** and **family cancer history** should be factored as mid-tier risk multipliers.  
- Age-related increase in **diabetes, blood pressure, and surgeries** amplifies risk; premiums should account for these clusters.

- **BMI Consideration:** Policyholders in the **normal BMI category** should be incentivized with lower premiums, while overweight/obese groups should face higher premiums due to increased likelihood of other health complications.  

- Offer **dynamic pricing**: Policyholders can lower premiums by enrolling in wellness programs, preventive check-ups, or fitness challenges.

- **Model Deployment:** For long-term scaling, start with **Random Forest** for transparency, then move toward **XGBoost** for efficient scaling in real-world applications.

- **Risk Management Advantage**: More granular pricing ensures you are not **underpricing high-risk customers** (which increases claim payouts) or **overpricing low-risk customers** (which pushes them to competitors).  
- Improves profitability while staying attractive to healthy customer segments.


### Future Work  
- Explore ensemble methods like **stacking** to build more sophisticated models and enable granular pricing strategies.  
- Implement **model monitoring** to track performance over time and detect **data drift**, ensuring stable and reliable predictions in production.

