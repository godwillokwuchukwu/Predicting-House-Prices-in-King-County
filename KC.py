import os
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import shap
import requests
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="King County ML Dashboard", layout="wide",
                   initial_sidebar_state="expanded")
st.title("King County House Price ML Dashboard")
st.markdown("**Created by Okwuchukwu Godwill Tochukwu/Data Scientist**")
st.caption("Built November 17, 2025")

# -------------------------
# AI key handling (user supplies their own)
# -------------------------
# We DO NOT provide keys. This app reads the key from env var or sidebar input.
ENV_KEY = os.getenv("AI_API_KEY", "").strip()

st.sidebar.header("AI Explanations (optional)")
ai_key_input = st.sidebar.text_input("Paste AI API key (optional)", value=ENV_KEY, type="password")
if ai_key_input:
    AI_API_KEY = ai_key_input
else:
    AI_API_KEY = ENV_KEY

def get_ai_explanation(prompt: str) -> str:
    """Best-effort hook for an AI provider. By default returns helpful fallback text when no key."""
    if not AI_API_KEY:
        return "AI explanation disabled — provide an API key in the sidebar or set AI_API_KEY environment variable."
    # Example placeholder for a simple OpenAI / other provider call — update as required for the API you use.
    try:
        headers = {"Authorization": f"Bearer {AI_API_KEY}"}
        payload = {"model":"gpt-4o", "messages":[{"role":"user","content":prompt}], "max_tokens":220}
        r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=12)
        if r.ok:
            return r.json()["choices"][0]["message"]["content"]
        return f"AI error: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"AI explanation unavailable ({e})"

# -------------------------
# Load & clean data
# -------------------------
@st.cache_data
def load_data(path="kc_house.csv"):
    df = pd.read_csv(path)
    # build date if Year/Month/Day present
    if set(['Year','Month','Day']).issubset(df.columns):
        df['date'] = pd.to_datetime(df[['Year','Month','Day']].astype(str).agg('-'.join, axis=1), errors='coerce')
        df = df.drop(['Year','Month','Day'], axis=1, errors='ignore')
    else:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['date'] = pd.NaT
    # normalize renovation year
    if 'yr_renovated' in df.columns and 'yr_built' in df.columns:
        df['yr_renovated'] = np.where(df['yr_renovated'] > 0, df['yr_renovated'], df['yr_built'])
    # common column fallbacks
    if 'sqft_lot' in df.columns and 'landsize' not in df.columns:
        df['landsize'] = df['sqft_lot']
    if 'sqft_lot15' in df.columns and 'landsize15' not in df.columns:
        df['landsize15'] = df['sqft_lot15']
    # numeric fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    # zipcode numeric encode if needed
    if 'zipcode' in df.columns and not np.issubdtype(df['zipcode'].dtype, np.number):
        df['zipcode'] = pd.to_numeric(df['zipcode'], errors='coerce').fillna(0).astype(int)
    return df

# allow user to upload or use local file
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload kc_house.csv (optional)", type=["csv"])
use_sample = False
if uploaded is not None:
    df = load_data(uploaded)
else:
    # default path fallback (developer: have a file in working dir)
    default_path = "kc_house.csv"
    if Path(default_path).exists():
        df = load_data(default_path)
    else:
        st.sidebar.info("No local kc_house.csv found. Upload one in the sidebar to use the app.")
        df = pd.DataFrame()  # empty placeholder

# -------------------------
# Features & target
# -------------------------
default_features = ['bedrooms', 'bathrooms', 'sqft_living', 'landsize', 'floors', 'waterfront', 'view',
                    'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'landsize15']
features = [f for f in default_features if f in df.columns] if not df.empty else []
target = 'price'

# -------------------------
# Utility helpers
# -------------------------
def train_rf_model(df_local, features_local, target_local):
    if target_local not in df_local.columns or not features_local:
        return None, None, None
    X = df_local[features_local].select_dtypes(include=[np.number]).fillna(0)
    y = pd.to_numeric(df_local[target_local], errors='coerce').fillna(0)
    if len(X) < 5:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    mdl.fit(X_train, y_train)
    return mdl, X_test, y_test

def safe_predict(mdl, input_df):
    try:
        return float(mdl.predict(input_df)[0])
    except Exception:
        return np.nan

def embed_shap_force_plot(explainer, shap_values, input_df, height=350):
    """
    Create JS SHAP force plot and embed in Streamlit using a temporary html file.
    shap.save_html works with force_plot objects.
    """
    try:
        shap.initjs()
        force_obj = shap.force_plot(explainer.expected_value, shap_values, input_df)
        # save to temp file and read
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        shap.save_html(tmp.name, force_obj)
        html = Path(tmp.name).read_text()
        st.components.v1.html(html, height=height, scrolling=True)
    except Exception as e:
        st.warning(f"Interactive SHAP failed ({e}). Attempting static fallback.")
        # fallback: simple bar of shap contributions if shap_values shape matches
        try:
            s = pd.Series(shap_values[0], index=input_df.columns).sort_values()
            fig, ax = plt.subplots(figsize=(8, max(2, len(s)*0.3)))
            s.plot.barh(ax=ax)
            ax.set_title("Approximate SHAP contributions")
            st.pyplot(fig)
        except Exception:
            st.write("SHAP unavailable — consider installing a compatible shap version.")

# insights generator: returns 5 insights & 5 recommendations for a given problem
def generate_insights(problem_id, local_df, model=None, features_local=None):
    insights = []
    recs = []
    # problem-specific heuristics
    if problem_id == 1:
        # Price distribution insights
        if 'price' in local_df.columns:
            q25, q50, q75 = np.percentile(local_df['price'], [25,50,75])
            mean = local_df['price'].mean()
            skew = local_df['price'].skew()
            insights = [
                f"Median price is ${q50:,.0f}; lower quartile ${q25:,.0f}, upper quartile ${q75:,.0f}.",
                f"Mean price ${mean:,.0f} vs median suggests skew: {skew:.2f}.",
                f"Top 5% homes have prices > ${np.percentile(local_df['price'],95):,.0f}.",
                f"Price distribution appears {'right' if skew>0 else 'left' if skew<0 else 'near-symmetric'}-skewed.",
                f"Significant spread — consider log-transform for regression modelling."
            ]
            recs = [
                "If using linear models, try log(price) to reduce skew.",
                "Examine top 5% houses for outliers or special features (waterfront, view).",
                "Use robust metrics (median) for dashboard KPIs to avoid outlier bias.",
                "Consider stratifying models by zipcode/grade to improve accuracy.",
                "Add feature-engineered interaction terms: sqft_living * grade, age * renovated_flag."
            ]
        else:
            insights = ["No price column available to compute insights."]
            recs = ["Upload a dataset with a numeric 'price' column."]
    elif problem_id == 2:
        # Renovation Impact
        if {'yr_built','yr_renovated','price'}.issubset(local_df.columns):
            local_df = local_df.copy()
            local_df['renovated'] = local_df['yr_renovated'] != local_df['yr_built']
            grp = local_df.groupby('renovated')['price'].agg(['count','mean','median'])
            insights = [
                f"Renovated homes: {int(grp.loc[True,'count']) if True in grp.index else 0}",
                f"Avg price renovated: ${grp.loc[True,'mean']:,.0f}" if True in grp.index else "No renovated homes",
                f"Avg price not renovated: ${grp.loc[False,'mean']:,.0f}" if False in grp.index else "No non-renovated homes",
                f"Median difference: ${abs(grp.loc[True,'median']-grp.loc[False,'median']):,.0f}" if set([True,False]).issubset(grp.index) else "",
                "Renovation effect varies by grade and zipcode."
            ]
            recs = [
                "Use matched samples (similar sqft/grade) to estimate renovation causal effect.",
                "Consider year-since-renovation as a feature to capture freshness.",
                "If renovation effect is small, focus on structural upgrades (kitchens, bathrooms).",
                "Segment by zipcode — renovation premium often varies geographically.",
                "Use hedonic regression controlling for grade/sqft to get adjusted premium."
            ]
        else:
            insights = ["Missing yr_built/yr_renovated/price columns for renovation analysis."]
            recs = ["Provide those columns in CSV."]
    elif problem_id == 3:
        # Neighborhood Clustering
        if {'zipcode','price'}.issubset(local_df.columns):
            med_by_zip = local_df.groupby('zipcode')['price'].median().sort_values()
            low = med_by_zip.head(3).index.tolist()
            high = med_by_zip.tail(3).index.tolist()
            insights = [
                f"Clusters separate neighborhoods — cheapest zips (median price): {low}.",
                f"Most expensive zips (median): {high}.",
                f"Cluster sizes may vary — inspect small clusters for niche markets.",
                "Clusters correlate strongly with zipcode and landsize.",
                "Use cluster labels as a feature in the prediction model to boost accuracy."
            ]
            recs = [
                "Target marketing to high-ARPU zip codes for premium listings.",
                "For low-price clusters consider smaller investment properties/affordable options.",
                "Create cluster-level dashboards (avg price, inventory) for agents.",
                "Investigate small clusters—those may represent specialized micro-markets.",
                "Use clustering as stratification when training models to reduce heterogeneity."
            ]
        else:
            insights = ["zipcode or price missing — cannot cluster neighborhoods."]
            recs = ["Add zipcode and price columns to dataset."]
    elif problem_id == 4:
        # Outlier Detection
        if 'price' in local_df.columns:
            q99 = np.percentile(local_df['price'], 99)
            out_count = (local_df['price'] > q99).sum()
            insights = [
                f"Top 1% of homes (price > ${q99:,.0f}) = {int(out_count)} homes.",
                "Outliers often have waterfront/view/very large sqft_living or data errors.",
                "Outliers can skew mean-based KPIs.",
                "Outlier flags can be used to remove extreme cases from training.",
                "Investigate outliers individually before deletion — some are valid premium homes."
            ]
            recs = [
                "Create filters for dashboards to exclude top 1% when showing averages.",
                "Manually inspect outliers for data quality (typos in price or sqft).",
                "If model performance is poor, try training excluding extreme outliers.",
                "Tag valid outliers (e.g., waterfront) instead of deleting.",
                "Use robust metrics (median, trimmed mean) for public-facing reports."
            ]
        else:
            insights = ["Price column missing — cannot detect outliers."]
            recs = ["Add price column."]
    elif problem_id == 5:
        # Forecast trend
        if 'date' in local_df.columns and 'price' in local_df.columns:
            ts = local_df.groupby('date')['price'].mean().dropna()
            if len(ts) > 10:
                slope = np.polyfit(np.arange(len(ts)), ts.values, 1)[0]
                insights = [
                    f"Observed trend slope ~ {slope:.2f} $/day over available dates.",
                    f"7-day rolling average reduces noise — useful for dashboards.",
                    "Short-term trend is sensitive to seasonality and sample size.",
                    "Large gaps or few dates reduce forecast reliability.",
                    "Use more advanced time-series models (Prophet/Auto-ARIMA) for better forecasts."
                ]
                recs = [
                    "Display both raw daily avg and 7-day rolling average on dashboards.",
                    "Flag forecast uncertainty (confidence intervals) for transparency.",
                    "If trend is weak, avoid overreacting to small daily changes.",
                    "Gather more frequent data for reliable short-term forecasts.",
                    "Consider monthly aggregation to remove high-frequency noise."
                ]
            else:
                insights = ["Insufficient date/price history to compute a stable trend."]
                recs = ["Provide longer time-series (daily/weekly) for forecasting."]
        else:
            insights = ["Missing date or price for trend analysis."]
            recs = ["Add those columns."]
    elif problem_id == 6:
        # Feature importance
        if model is not None and features_local:
            fi = pd.Series(model.feature_importances_, index=features_local).sort_values(ascending=False)
            top3 = fi.head(3).index.tolist()
            insights = [
                f"Top features: {', '.join(top3)}.",
                f"Feature importance skew: top features contribute ~{fi.head(3).sum():.2f} of total importance.",
                "Some numeric features may hide non-linear relationships.",
                "Feature interactions (sqft_living x grade) may be important.",
                "Low-importance features could be dropped to speed up models."
            ]
            recs = [
                "Focus data collection on the top features to improve model performance.",
                "Engineer interactions and polynomial terms for top features.",
                "Consider SHAP for local explanations on key predictions.",
                "Prune very low importance features when deploying to edge devices.",
                "Retrain model with cross-validation to confirm importance stability."
            ]
        else:
            insights = ["Model not trained or features missing for importance analysis."]
            recs = ["Train a model or provide numeric features."]
    elif problem_id == 7:
        # Grade classification
        if 'grade' in local_df.columns:
            grades = local_df['grade'].value_counts().sort_index()
            insights = [
                f"Grades frequency shows common grades: {grades.head(3).index.tolist()}.",
                "Classification accuracy depends on number of features and their quality.",
                "Grades are coarse ordinal indicators of construction quality.",
                "Some grades may be rare — consider grouping rare classes.",
                "Tree-based models perform well on ordinal-ish targets."
            ]
            recs = [
                "Group sparse grade classes to improve classifier stability.",
                "Provide photo or inspection features to improve grade predictions.",
                "Use class-weighting if class imbalance is severe.",
                "Present predicted probabilities, not only labels, to users.",
                "Validate classifier on a hold-out set or cross-validation."
            ]
        else:
            insights = ["Grade column missing."]
            recs = ["Add 'grade' column to dataset."]
    elif problem_id == 8:
        # Similar houses
        if features_local:
            insights = [
                "Similar-houses search uses nearest neighbors on numeric features.",
                "Neighbors often share sqft, bedrooms, and zipcode.",
                "Using scaled features improves neighbor quality.",
                "Use more neighbors (5-10) for more robust suggestions.",
                "Filter neighbors by recent sales date to keep comparisons fresh."
            ]
            recs = [
                "Scale features and include age/renovation flags for better matches.",
                "Allow users to filter neighbors by date or zipcode.",
                "Show price per sqft for more direct comparisons.",
                "Limit the neighbor set to same property type to avoid mismatches.",
                "Cache neighbor indices for performance on large datasets."
            ]
        else:
            insights = ["No numeric features to compute similarity."]
            recs = ["Provide numeric features."]
    elif problem_id == 9:
        # Waterfront premium
        if 'waterfront' in local_df.columns and 'price' in local_df.columns:
            wf = local_df[local_df['waterfront']==1]
            non_wf = local_df[local_df['waterfront']==0]
            if len(wf)>0 and len(non_wf)>0:
                premium = wf['price'].mean() - non_wf['price'].mean()
                insights = [
                    f"Avg waterfront price: ${wf['price'].mean():,.0f}; non-waterfront: ${non_wf['price'].mean():,.0f}.",
                    f"Estimated premium ≈ ${premium:,.0f}.",
                    "Premium varies by zipcode and view score.",
                    "Small sample of waterfront homes can inflate estimates.",
                    "Consider controlling for sqft and grade when estimating premium."
                ]
                recs = [
                    "Report adjusted premium controlling for sqft/grade using regression.",
                    "If premium large, create waterfront-focused marketing.",
                    "Validate premium with matched-pair samples if possible.",
                    "Display sample size and uncertainty for premium estimate.",
                    "Use model-based counterfactuals to estimate premium per property."
                ]
            else:
                insights = ["Insufficient waterfront or non-waterfront samples."]
                recs = ["Need both waterfront and non-waterfront rows to compute premium."]
        else:
            insights = ["waterfront or price missing."]
            recs = ["Add both columns."]
    elif problem_id == 10:
        # Condition segments
        if {'condition','price'}.issubset(local_df.columns):
            segs = KMeans(n_clusters=3, random_state=42).fit_predict(local_df[['condition','price']].select_dtypes(include=[np.number]).fillna(0))
            insights = [
                "KMeans created 3 condition-price segments.",
                "Segments likely represent low-, mid-, and high-value groups.",
                "Segments can map to marketing tiers or renovation needs.",
                "Check segment sizes to ensure balanced groups.",
                "Segment centroids reveal price vs condition tradeoffs."
            ]
            recs = [
                "Create tailored recommendations for each segment (e.g., renovate vs sell).",
                "Use segments to prioritize inspections and lead routing.",
                "Combine with zipcode to find high-opportunity segments.",
                "Visualize centroid values in a small table for quick interpretation.",
                "Recompute segments periodically to capture market shifts."
            ]
        else:
            insights = ["condition or price missing."]
            recs = ["Add both columns."]
    else:
        insights = ["No insights available for this analysis."]
        recs = ["Select a valid problem."]
    # ensure 5 items each
    insights = (insights + [""]*5)[:5]
    recs = (recs + [""]*5)[:5]
    return insights, recs

# -------------------------
# Train main model once (if possible)
# -------------------------
if not df.empty and features and target in df.columns:
    model, X_test, y_test = train_rf_model(df, features, target)
else:
    model, X_test, y_test = None, None, None

# -------------------------
# Layout: Tabs for problems
# -------------------------
tabs = st.tabs([
    "Overview", "1. Price Prediction", "2. Renovation Impact",
    "3. Neighborhood Clustering", "4. Outlier Detection", "5. Trend Forecast",
    "6. Feature Importance", "7. Grade Classification", "8. Similar Houses",
    "9. Waterfront Premium", "10. Condition Segments"
])

# ---- Overview tab ----
with tabs[0]:
    st.header("Overview & KPIs")
    if df.empty:
        st.warning("No dataset loaded. Upload kc_house.csv via the sidebar to continue.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Price", f"${df['price'].mean():,.0f}" if 'price' in df.columns else "N/A")
        col2.metric("Total Houses", f"{len(df):,}")
        col3.metric("Avg Living Area", f"{df['sqft_living'].mean():,.0f} sqft" if 'sqft_living' in df.columns else "N/A")
        col4.metric("Waterfront Homes", int(df['waterfront'].sum()) if 'waterfront' in df.columns else 0)

        st.markdown("### Price distribution")
        if 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(9,3))
            sns.histplot(df['price'], kde=True, ax=ax)
            st.pyplot(fig)
            insights, recs = generate_insights(1, df)
            st.markdown("**Top 5 insights**")
            for i, s in enumerate(insights,1):
                st.write(f"{i}. {s}")
            st.markdown("**Top 5 recommendations**")
            for i, r in enumerate(recs,1):
                st.write(f"{i}. {r}")
        else:
            st.info("Upload a dataset with a numeric 'price' column to see distribution.")

# ---- Problem 1: Price Prediction ----
with tabs[1]:
    st.header("1. House Price Prediction (with interactive SHAP)")
    if df.empty or not features:
        st.info("Dataset or numeric features missing. Upload kc_house.csv in the sidebar.")
    else:
        with st.form("predict_form", clear_on_submit=False):
            st.subheader("Enter house features")
            # arrange features in 3 columns for a clean form
            cols = st.columns(3)
            input_vals = {}
            for i, feat in enumerate(features):
                with cols[i % 3]:
                    default = int(round(df[feat].mean())) if np.issubdtype(df[feat].dtype, np.integer) else float(df[feat].mean())
                    if feat in ['bedrooms','bathrooms','floors','waterfront','view','condition','grade','yr_built','yr_renovated','zipcode']:
                        input_vals[feat] = st.number_input(feat.replace('_',' ').title(), value=int(default), step=1, key=f"p1_{feat}")
                    else:
                        input_vals[feat] = st.number_input(feat.replace('_',' ').title(), value=float(default), step=50.0, format="%.0f", key=f"p1_{feat}")
            predict_btn = st.form_submit_button("Predict")

        if predict_btn:
            input_df = pd.DataFrame([input_vals])[features].select_dtypes(include=[np.number]).fillna(0)
            pred = safe_predict(model, input_df) if model is not None else np.nan
            if np.isnan(pred):
                st.error("Model not available or prediction failed. Ensure dataset has adequate numeric features and model trained.")
            else:
                st.success(f"Predicted price: **${pred:,.0f}**")
                # interactive SHAP
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)
                    st.markdown("**Interactive SHAP explanation (force plot)**")
                    embed_shap_force_plot(explainer, shap_values, input_df, height=350)
                except Exception as e:
                    st.warning(f"SHAP interactive plot failed ({e}). Showing static feature importances.")
                    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
                    fig, ax = plt.subplots(figsize=(8,4))
                    fi.plot.barh(ax=ax)
                    st.pyplot(fig)

                # Price distribution + insights & recs
                fig, ax = plt.subplots(figsize=(9,3))
                sns.histplot(df['price'], kde=True, ax=ax)
                ax.axvline(pred, color="red", linestyle="--", linewidth=2, label="Prediction")
                ax.legend()
                st.pyplot(fig)

                insights, recs = generate_insights(1, df)
                st.markdown("**Insights**")
                for i,s in enumerate(insights,1): st.write(f"{i}. {s}")
                st.markdown("**Recommendations**")
                for i,r in enumerate(recs,1): st.write(f"{i}. {r}")

                # optional AI explanation
                st.markdown("**AI Explanation (optional)**")
                st.write(get_ai_explanation(f"Explain why this house is worth ${pred:,.0f} given features: {input_vals}"))

# ---- Problem 2: Renovation Impact ----
with tabs[2]:
    st.header("2. Renovation Impact")
    if df.empty:
        st.info("No data.")
    else:
        if {'yr_built','yr_renovated','price'}.issubset(df.columns):
            df2 = df.copy()
            df2['renovated'] = df2['yr_renovated'] != df2['yr_built']
            fig, ax = plt.subplots(figsize=(8,4))
            sns.boxplot(x='renovated', y='price', data=df2, ax=ax, palette="Set2")
            ax.set_xlabel("Renovated")
            ax.set_title("Price: Renovated vs Not Renovated")
            st.pyplot(fig)
            insights, recs = generate_insights(2, df2)
            st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
            st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]
            st.write(get_ai_explanation("Do renovated houses cost more on average?"))
        else:
            st.info("Required columns for renovation analysis missing (yr_built, yr_renovated, price).")

# ---- Problem 3: Neighborhood Clustering ----
with tabs[3]:
    st.header("3. Neighborhood Clustering")
    if df.empty or 'price' not in df.columns:
        st.info("Upload dataset with price and zipcode for clustering.")
    else:
        cluster_cols = [c for c in ['zipcode','price','landsize'] if c in df.columns]
        if not cluster_cols:
            st.info("Need zipcode/price/landsize to cluster.")
        else:
            Xc = StandardScaler().fit_transform(df[cluster_cols].select_dtypes(include=[np.number]).fillna(0))
            n_clusters = st.slider("Number of clusters", 2, 8, 5)
            clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(Xc)
            df['cluster'] = clusters
            chart = alt.Chart(df.sample(min(1000,len(df)))).mark_circle(size=60).encode(
                x='zipcode:O' if 'zipcode' in df.columns else alt.X('index:O'),
                y='price:Q',
                color='cluster:N',
                tooltip=['zipcode','price','grade'] if {'zipcode','price','grade'}.issubset(df.columns) else ['price']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            insights, recs = generate_insights(3, df)
            st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
            st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]

# ---- Problem 4: Outlier Detection ----
with tabs[4]:
    st.header("4. Outlier Detection")
    if df.empty or 'price' not in df.columns:
        st.info("Upload dataset with numeric columns.")
    else:
        contamination = st.slider("Outlier contamination (fraction)", 0.001, 0.05, 0.01)
        iso = IsolationForest(contamination=float(contamination), random_state=42)
        numeric_for_iso = df[features].select_dtypes(include=[np.number]).fillna(0) if features else df.select_dtypes(include=[np.number]).fillna(0)
        if numeric_for_iso.shape[1] == 0:
            st.info("No numeric features found for outlier detection.")
        else:
            df['outlier'] = iso.fit_predict(numeric_for_iso)
            outliers = df[df['outlier']==-1]
            st.write(f"Found **{len(outliers)}** outliers (flag = -1).")
            fig, ax = plt.subplots(figsize=(8,4))
            if 'sqft_living' in df.columns and 'price' in df.columns:
                sns.scatterplot(data=df, x='sqft_living', y='price', hue='outlier', palette={1:"lightblue",-1:"red"}, ax=ax, alpha=0.8)
            else:
                sns.scatterplot(data=df.reset_index().sample(min(1000,len(df))), x='index', y=df.columns[0], hue='outlier', ax=ax)
            st.pyplot(fig)
            insights, recs = generate_insights(4, df)
            st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
            st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]

# ---- Problem 5: Trend Forecast ----
with tabs[5]:
    st.header("5. Price Trend Forecast")
    if df.empty or 'date' not in df.columns or 'price' not in df.columns:
        st.info("Upload dataset with date and price columns.")
    else:
        ts = df.groupby('date')['price'].mean().sort_index()
        if len(ts) < 10:
            st.warning("Insufficient date history for a stable trend.")
        else:
            # 7-day average + linear trend
            ts_df = ts.to_frame(name='price')
            ts_df['7d_avg'] = ts_df['price'].rolling(7, min_periods=1).mean()
            ts_df['days'] = (ts_df.index - ts_df.index.min()).days
            lr = LinearRegression().fit(ts_df[['days']], ts_df['price'])
            future_days = 30
            future = pd.DataFrame({'days': np.arange(ts_df['days'].max()+1, ts_df['days'].max()+future_days+1)})
            forecast_dates = pd.date_range(ts_df.index.max()+pd.Timedelta(days=1), periods=future_days)
            fig, ax = plt.subplots(figsize=(11,4))
            ax.plot(ts_df.index, ts_df['price'], alpha=0.4, label="Daily avg")
            ax.plot(ts_df.index, ts_df['7d_avg'], linewidth=2, label="7-day avg")
            ax.plot(forecast_dates, lr.predict(future), "--", linewidth=2, label="30-day linear forecast")
            ax.set_ylabel("Average price")
            ax.legend()
            st.pyplot(fig)
            insights, recs = generate_insights(5, df)
            st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
            st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]

# ---- Problem 6: Feature Importance ----
with tabs[6]:
    st.header("6. Feature Importance")
    if model is None or not features:
        st.info("Model not trained (needs numeric features).")
    else:
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(3, len(fi)*0.25)))
        fi.plot.barh(ax=ax)
        ax.set_title("Random Forest feature importances")
        st.pyplot(fig)
        insights, recs = generate_insights(6, df, model, features)
        st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
        st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]

# ---- Problem 7: Grade Classification ----
with tabs[7]:
    st.header("7. Grade Classification")
    if df.empty or 'grade' not in df.columns:
        st.info("Dataset missing 'grade'.")
    else:
        Xg = df.drop(['price','grade','date'], axis=1, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        if Xg.shape[1] == 0:
            st.info("No numeric features to train classifier.")
        else:
            clf = DecisionTreeClassifier(max_depth=10, random_state=42)
            Xtr, Xte, ytr, yte = train_test_split(Xg, df['grade'], test_size=0.2, random_state=42)
            clf.fit(Xtr, ytr)
            acc = accuracy_score(yte, clf.predict(Xte))
            st.success(f"Classification accuracy: **{acc:.1%}**")
            insights, recs = generate_insights(7, df)
            st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
            st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]

# ---- Problem 8: Similar Houses ----
with tabs[8]:
    st.header("8. Find Similar Houses")
    if df.empty or not features:
        st.info("Dataset or numeric features missing.")
    else:
        idx = st.selectbox("Select house index (first 500)", df.index[:500])
        k = st.slider("Number of similar houses to show", 1, 10, 5)
        nn = NearestNeighbors(n_neighbors=k+1)
        X_nn = df[features].select_dtypes(include=[np.number]).fillna(0)
        nn.fit(X_nn)
        _, inds = nn.kneighbors(X_nn.loc[[idx]])
        similar = df.iloc[inds[0][1:]]
        st.dataframe(similar[['price','bedrooms','bathrooms','sqft_living','grade','zipcode']].reset_index(drop=True).round(0))
        insights, recs = generate_insights(8, df, model, features)
        st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
        st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]

# ---- Problem 9: Waterfront Premium ----
with tabs[9]:
    st.header("9. Waterfront Premium")
    if df.empty or 'waterfront' not in df.columns or 'price' not in df.columns:
        st.info("Dataset missing waterfront or price.")
    else:
        if df['waterfront'].sum() == 0:
            st.info("No waterfront properties present in dataset.")
        else:
            wf = df[df['waterfront']==1]
            non_wf = df[df['waterfront']==0]
            if len(wf)>0 and len(non_wf)>0:
                premium = wf['price'].mean() - non_wf['price'].mean()
                st.metric("Estimated Waterfront Premium", f"${premium:,.0f}", delta=f"${(premium / non_wf['price'].mean() * 100):.1f}%")
                fig, ax = plt.subplots(figsize=(8,4))
                sns.boxplot(x='waterfront', y='price', data=df, ax=ax, palette="Set1")
                st.pyplot(fig)
                insights, recs = generate_insights(9, df)
                st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
                st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]
            else:
                st.info("Insufficient waterfront / non-waterfront samples.")

# ---- Problem 10: Condition Segments ----
with tabs[10]:
    st.header("10. Condition Segments")
    if df.empty or not {'condition','price'}.issubset(df.columns):
        st.info("Missing condition or price columns.")
    else:
        Xc = df[['condition','price']].select_dtypes(include=[np.number]).fillna(0)
        n_seg = st.slider("Number of condition segments", 2, 5, 3)
        seg_labels = KMeans(n_clusters=n_seg, random_state=42).fit_predict(Xc)
        df['segment'] = seg_labels
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(x='segment', y='price', data=df, palette="Set3", ax=ax)
        st.pyplot(fig)
        insights, recs = generate_insights(10, df)
        st.markdown("**Insights**"); [st.write(f"{i+1}. {s}") for i,s in enumerate(insights)]
        st.markdown("**Recommendations**"); [st.write(f"{i+1}. {r}") for i,r in enumerate(recs)]

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Tip: Provide an AI API key in the sidebar to enable richer, model-powered explanations.")
