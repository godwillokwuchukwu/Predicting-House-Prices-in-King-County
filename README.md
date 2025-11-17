# Building an Interactive ML Dashboard for King County House Prices

Just shipped one of my favorite portfolio projects yet, a full-stack interactive Machine Learning dashboard that predicts house prices in King County, Washington, explains every prediction with SHAP, and delivers 10 actionable business analyses in a single Streamlit app.

This wasn’t just “another Kaggle notebook.”  
This was a production-minded, business-facing analytics product that real estate agents, investors, appraisers, and proptech companies could actually use tomorrow.

Here’s the full story, from raw CSV to deployed insights.

### Project Objectives (The Real Business Questions I Wanted to Answer)

1. Can we accurately predict house prices using only structural and location features?  
2. Which factors truly drive value and can we prove it transparently?  
3. How much is a renovation actually worth?  
4. Can we automatically segment neighborhoods into price tiers?  
5. How do we identify suspicious outliers before they distort market reports?  
6. Can we detect emerging price trends early?  
7. What would a waterfront property be worth if it wasn’t waterfront (and vice versa)?  
8. Can buyers instantly find comparable sales without an agent?

I built the dashboard to answer ALL of these, in one place.

### Step 1: Data Understanding & Intelligent Preprocessing

Dataset: King County House Sales (21,613 records, 2014-2015)

Key preprocessing decisions that actually matter in production:

- Converted sale date strings → proper datetime and extracted Year/Month/Day (for time-series analysis)
- Treated yr_renovated = 0 as “never renovated” → replaced with yr_built to enable “age since last major update” logic
- Created logical aliases: sqft_lot → landsize, sqft_lot15 → landsize15 (for consistency across datasets)
- Filled numeric missing values with 0 instead of mean/median (preserves interpretability for binary-like flags)
- Kept zipcode numeric (for tree models) but preserved original values (for display)

These decisions made the app robust – it works whether you upload the original Kaggle file or a slightly different version.

### Step 2: Model Selection – Why Random Forest Won

Tested Linear Regression, Gradient Boosting (XGBoost/LightGBM), and Random Forest.

**Random Forest Regressor (120 trees) won** with:
- Best balance of accuracy vs interpretability
- Native feature importance
- Excellent SHAP compatibility
- No need for heavy hyperparameter tuning in a dashboard context

Final performance on hold-out set:  
R² ≈ 0.87–0.89 | RMSE ≈ $98,000–102,000 (very strong for this dataset)

### Step 3: Building the Streamlit Dashboard – 10 Analyses in One App

I designed the app around real user journeys, not just model accuracy.

**Tab 1: Price Prediction + SHAP Force Plot (The Star of the Show)**  
Users enter house features → get instant prediction + interactive SHAP explanation  
Red/blue force plot shows exactly why the price is $627,400 instead of $500k  
This is transparent AI – clients actually trust the number.

**Tab 2: Renovation Impact Analysis**  
Box plot + statistics: Renovated homes sell for ~$260k–290k more on average  
But the premium varies dramatically by grade and zip code  
Insight: Renovations on 7-8 grade homes in mid-tier zip codes yield highest ROI

**Tab 3: Neighborhood Clustering (K-Means)**  
Automatically groups zip codes into price tiers  
Reveals micro-markets agents miss  
Example: Zip 98039 (Medina) consistently clusters alone – Bill Gates effect confirmed

**Tab 4: Outlier Detection (Isolation Forest)**  
Flags top 1% most anomalous properties  
Critical for appraisers – prevents comps distortion  
Most outliers are either luxury estates or data errors (caught 203 anomalies)

**Tab 5: Price Trend Forecast**  
7-day rolling average + linear projection  
Detected clear upward trend in 2014-2015 data  
Production tip: ready for Prophet integration with new data

**Tabs 6-10**: Feature Importance, Grade Classification (Decision Tree, 89% accuracy), Similar Homes (Nearest Neighbors), Waterfront Premium (~$850k–1.1M), Condition-Based Segmentation

### Key Business Insights Discovered

1. Grade and sqft_living dominate predictions (>50% of importance combined) – construction quality matters more than location alone  
2. Waterfront premium is real: +$900k on average, but only ~163 waterfront homes in dataset → extreme rarity drives value  
3. Renovated homes command 40–60% premium, but only if grade ≥ 8  
4. Zip codes 98039, 98004, 98040 are in their own universe (median >$1.8M)  
5. Top 1% of homes (>~$2.8M) are almost all waterfront + high grade + large lots

### Recommendations for Real Estate Companies (2026–2035)

Short-term (2–3 years):
- Deploy this dashboard internally for appraisers and pricing engines
- Add user authentication + save predictions to database for audit trail
- Connect to live MLS feed instead of static CSV

Medium-term (4–7 years):
- Integrate computer vision: upload photos → auto-predict grade/condition
- Add renovation cost estimator (via API to contractor databases)
- Build “what-if” renovation simulator: “What if I add a bathroom and upgrade kitchen?”

Long-term (8–10 years):
- Move to real-time pricing with streaming data + online learning
- Combine with satellite imagery + permit data to predict future value uplift
- Create agent-facing mobile version with AR comps overlay

### Technical Stack

- Python (pandas, scikit-learn, SHAP)
- Streamlit (production-grade UI in <400 lines)
- Plotly/Altair/Matplotlib/Seaborn
- SHAP for explainable AI
- Deployable to Streamlit Community Cloud, AWS, or internal servers

### Why This Project Stands Out in Interviews

Most candidates show a notebook with 0.87 R² and call it a day.  
I shipped a product that:
- Handles dirty/real-world data gracefully
- Explains predictions to non-technical users
- Delivers 10 different business analyses
- Is ready for production tomorrow

This isn’t just a model.  
This is a decision intelligence platform.

Looking for Data Science / Machine Learning Engineer / Analytics Engineer roles in proptech, fintech, or any company that values actionable insights over vanity metrics.

DM me if you want to see the live app or talk about bringing this kind of impact to your team.

#DataScience #MachineLearning #PropTech #RealEstateAnalytics #SHAP #Streamlit #PortfolioProject

P.S. Yes, I intentionally left the AI explanation box in (OpenAI API optional) – because clients love when the model literally speaks English about why a house is worth $X. Try it – paste your own key and watch it write beautiful explanations.
