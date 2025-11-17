# kc_house_ml_app.py
# King County House Price ML Dashboard — fixed & more robust

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import altair as alt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# suppress harmless warnings for cleaner UI
warnings.filterwarnings("ignore")

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(page_title="King County ML Dashboard", layout="wide")
st.title("King County House Price ML Dashboard")
st.markdown("**10 Real-World ML Problems • SHAP • Charts • AI Explanations • Made in Nigeria**")

# =============================================
# OPTIONAL GROK AI (Works without key)
# =============================================
GROK_API_KEY = ""  # Paste your key here to enable AI explanations

def get_ai_explanation(prompt: str) -> str:
    if not GROK_API_KEY or GROK_API_KEY.strip() == "":
        return "AI explanation disabled – add your Grok API key to enable."
    try:
        headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
        payload = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 220,
            "temperature": 0.3
        }
        r = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=12)
        if r.ok:
            return r.json()["choices"][0]["message"]["content"]
        return "AI error: " + r.text[:200]
    except Exception as e:
        return f"AI explanation unavailable ({e})"

# =============================================
# LOAD & CLEAN DATA
# =============================================
@st.cache_data
def load_data(path="kc_house.csv"):
    df = pd.read_csv(path)
    # If Year/Month/Day present build a date, else try existing date column
    if set(['Year','Month','Day']).issubset(df.columns):
        try:
            df['date'] = pd.to_datetime(df[['Year','Month','Day']].astype(str).agg('-'.join, axis=1))
            df = df.drop(['Year', 'Month', 'Day'], axis=1)
        except Exception:
            # fallback: try any existing date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            # create a dummy date if none exists
            df['date'] = pd.Timestamp("2000-01-01")

    # normalize renovation year
    if 'yr_renovated' in df.columns and 'yr_built' in df.columns:
        df['yr_renovated'] = np.where(df['yr_renovated'] > 0, df['yr_renovated'], df['yr_built'])
    # ensure int types where reasonable
    for col in ['yr_built','yr_renovated']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # some datasets use different column names for lot size / sqft_lot — try to standardize a couple
    if 'sqft_lot' in df.columns and 'landsize' not in df.columns:
        df['landsize'] = df['sqft_lot']
    if 'sqft_lot15' in df.columns and 'landsize15' not in df.columns:
        df['landsize15'] = df['sqft_lot15']

    # fill na for numeric features with 0 for safety (better: impute properly)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df

df = load_data()

# =============================================
# FEATURES & TARGET — fixed typo and defensive checks
# =============================================
# original expected features (adjust names if your CSV differs)
features = ['bedrooms', 'bathrooms', 'sqft_living', 'landsize', 'floors', 'waterfront', 'view',
            'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'landsize15']
# keep only features that actually exist and are numeric
available_features = [f for f in features if f in df.columns]
# If zipcode exists as string, convert to numeric code (models need numeric)
if 'zipcode' in available_features and not np.issubdtype(df['zipcode'].dtype, np.number):
    try:
        df['zipcode'] = pd.to_numeric(df['zipcode'], errors='coerce').fillna(0).astype(int)
    except:
        df['zipcode'] = df['zipcode'].astype('category').cat.codes
# final features used
features = [f for f in available_features]
target = 'price'

# =============================================
# KPI CARDS
# =============================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Price", f"${df[target].mean():,.0f}" if target in df.columns else "N/A")
c2.metric("Total Houses", f"{len(df):,}")
c3.metric("Avg Living Area", f"{df['sqft_living'].mean():,.0f} sqft" if 'sqft_living' in df.columns else "N/A")
c4.metric("Waterfront Homes", int(df['waterfront'].sum()) if 'waterfront' in df.columns else 0)

# =============================================
# SIDEBAR
# =============================================
problem = st.sidebar.selectbox("Choose ML Problem", [
    "1. House Price Prediction",
    "2. Renovation Impact",
    "3. Neighborhood Clustering",
    "4. Outlier Detection",
    "5. Price Trend Forecast",
    "6. Feature Importance",
    "7. Grade Classification",
    "8. Similar Houses",
    "9. Waterfront Premium",
    "10. Condition Segments"
])

# =============================================
# TRAIN MAIN MODEL (safe — uses numeric features only)
# =============================================
@st.cache_resource
def train_model(df_local, features_local, target_local):
    if target_local not in df_local.columns or len(features_local) == 0:
        return None, None, None
    X = df_local[features_local].select_dtypes(include=[np.number]).fillna(0)
    y = pd.to_numeric(df_local[target_local], errors='coerce').fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(df, features, target)

# helper: safe predict
def safe_predict(mdl, input_df):
    try:
        return float(mdl.predict(input_df)[0])
    except Exception:
        return np.nan

# =============================================
# PROBLEM 1: House Price Prediction
# =============================================
if problem == "1. House Price Prediction":
    st.header("Predict House Price")
    if not features or target not in df.columns:
        st.error("Required features or target column missing from dataset. Check your CSV column names.")
    else:
        with st.form("input_form"):
            inputs = {}
            cols = st.columns(3)
            # use reasonable defaults from dataframe means/medians
            for i, feat in enumerate(features):
                with cols[i % 3]:
                    col_mean = df[feat].mean() if feat in df.columns else 0
                    if feat in ['bedrooms','bathrooms','floors','waterfront','view','condition','grade','yr_built','yr_renovated','zipcode']:
                        inputs[feat] = st.number_input(feat.replace('_',' ').title(), value=int(round(col_mean)), step=1)
                    else:
                        inputs[feat] = st.number_input(feat.replace('_',' ').title(), value=float(round(col_mean, 0)), step=50.0, format="%.0f")
            submitted = st.form_submit_button("Predict Price")

        if submitted:
            input_df = pd.DataFrame([inputs])[features].select_dtypes(include=[np.number]).fillna(0)
            pred = safe_predict(model, input_df)
            if np.isnan(pred):
                st.error("Prediction failed — check model/features.")
            else:
                st.success(f"**Predicted Price: ${pred:,.0f}**")

                # SHAP explanation (best-effort)
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(input_df)
                    # try a matplotlib-friendly visualization
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 3))
                    # build a simple bar of per-feature contribution using shap values if shape matches
                    if isinstance(shap_vals, np.ndarray) and shap_vals.shape[1] == input_df.shape[1]:
                        contrib = pd.Series(shap_vals[0], index=input_df.columns).sort_values()
                        contrib.plot.barh(ax=ax_shap)
                        ax_shap.set_title("SHAP feature contributions (approx)")
                        st.pyplot(fig_shap)
                    else:
                        # fallback waterfall if shap returns an object
                        try:
                            shap.plots.bar(explainer(input_df), show=False)
                            st.pyplot(bbox_inches='tight')
                        except Exception:
                            st.write("SHAP produced unexpected format; showing feature importances instead.")
                            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
                            fi.plot.barh(figsize=(8,4))
                            st.pyplot(plt.gcf())
                except Exception as e:
                    # shap not installed or failed — show feature importances
                    st.info("SHAP unavailable; showing model feature importances.")
                    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
                    fig, ax = plt.subplots(figsize=(8,4))
                    sns.barplot(x=fi.values, y=fi.index, ax=ax)
                    ax.set_title("Feature importances (Random Forest)")
                    st.pyplot(fig)

                # AI Explanation
                st.markdown("**AI Explanation**")
                st.write(get_ai_explanation(f"In simple English, why is this house worth ${pred:,.0f}?"))

                # Price Distribution
                fig, ax = plt.subplots()
                sns.histplot(df[target], kde=True, ax=ax)
                ax.axvline(pred, color="red", linestyle="--", linewidth=3, label="Your Prediction")
                ax.legend()
                st.pyplot(fig)

# =============================================
# OTHER 9 PROBLEMS
# =============================================
elif problem == "2. Renovation Impact":
    st.header("Do Renovations Add Value?")
    if {'yr_renovated','yr_built', 'price'}.issubset(df.columns):
        df = df.copy()
        df['renovated'] = df['yr_renovated'] != df['yr_built']
        fig, ax = plt.subplots()
        sns.boxplot(x='renovated', y='price', data=df, palette="Set2", ax=ax)
        ax.set_title("Price: Renovated vs Not Renovated")
        st.pyplot(fig)
        st.write(get_ai_explanation("Do renovated houses cost more on average?"))
    else:
        st.info("Required columns for this analysis are missing (yr_built, yr_renovated, price).")

elif problem == "3. Neighborhood Clustering":
    st.header("Neighborhood Price Clusters")
    cluster_cols = [c for c in ['zipcode', 'price', 'landsize'] if c in df.columns]
    if cluster_cols:
        Xc = StandardScaler().fit_transform(df[cluster_cols].select_dtypes(include=[np.number]).fillna(0))
        df['cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(Xc)
        chart = alt.Chart(df).mark_circle(size=80).encode(
            x=alt.X('zipcode:O' if 'zipcode' in df.columns else 'index:O'),
            y='price:Q',
            color='cluster:N',
            tooltip=['zipcode','price','grade'] if {'zipcode','price','grade'}.issubset(df.columns) else ['price']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Needed columns (zipcode, price, landsize) are missing.")

elif problem == "4. Outlier Detection":
    st.header("Outlier Detection")
    if len(features) == 0:
        st.info("No numeric features available for outlier detection.")
    else:
        iso = IsolationForest(contamination=0.01, random_state=42)
        X_iso = df[features].select_dtypes(include=[np.number]).fillna(0)
        df['outlier'] = iso.fit_predict(X_iso)
        outliers = df[df['outlier'] == -1]
        st.write(f"**{len(outliers)} extreme outliers found**")
        fig, ax = plt.subplots()
        if 'sqft_living' in df.columns and 'price' in df.columns:
            sns.scatterplot(data=df, x='sqft_living', y='price', hue='outlier', 
                            palette={1:"lightblue", -1:"red"}, alpha=0.8, ax=ax)
        else:
            sns.scatterplot(data=df.reset_index(), x='index', y=df.columns[0], hue='outlier', ax=ax)
        st.pyplot(fig)

elif problem == "5. Price Trend Forecast":
    st.header("30-Day Price Trend Forecast")
    if 'date' in df.columns and 'price' in df.columns:
        ts = df.groupby('date')['price'].mean().sort_index().to_frame()
        ts['7d_avg'] = ts['price'].rolling(7, min_periods=1).mean()
        ts['days'] = (ts.index - ts.index.min()).days
        trend = LinearRegression().fit(ts[['days']], ts['price'])
        future = pd.DataFrame({'days': np.arange(ts['days'].max()+1, ts['days'].max()+31)})
        forecast_dates = pd.date_range(ts.index.max()+pd.Timedelta(days=1), periods=30)
        fig, ax = plt.subplots(figsize=(11,5))
        ax.plot(ts.index, ts['price'], alpha=0.5, label="Daily Avg")
        ax.plot(ts.index, ts['7d_avg'], linewidth=2.5, label="7-Day Average")
        ax.plot(forecast_dates, trend.predict(future), "--", linewidth=2.5, label="30-Day Forecast")
        ax.set_ylabel("Average Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.write(f"**Daily Trend: ${trend.coef_[0]:,.0f}/day**")
    else:
        st.info("date and price columns required for trend forecast.")

elif problem == "6. Feature Importance":
    st.header("What Drives House Prices?")
    if model is not None and len(features) > 0:
        imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=imp.values, y=imp.index, ax=ax)
        ax.set_title("Top 10 Most Important Features")
        st.pyplot(fig)
    else:
        st.info("Model not trained or features missing.")

elif problem == "7. Grade Classification":
    st.header("Predict House Grade (1–13)")
    if 'grade' in df.columns:
        Xg = df.drop(['price','grade','date'], axis=1, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        yg = df['grade']
        if Xg.shape[1] == 0:
            st.info("No numeric features available to train classifier.")
        else:
            clf = DecisionTreeClassifier(max_depth=10, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(Xg, yg, test_size=0.2, random_state=42)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            st.success(f"**Classification Accuracy: {acc:.1%}**")
    else:
        st.info("Column 'grade' not available in dataset.")

elif problem == "8. Similar Houses":
    st.header("Find Similar Houses")
    if len(features) == 0:
        st.info("No features available to compute similarity.")
    else:
        idx = st.selectbox("Select house index", df.index[:500])
        nn = NearestNeighbors(n_neighbors=6)
        X_nn = df[features].select_dtypes(include=[np.number]).fillna(0)
        nn.fit(X_nn)
        _, indices = nn.kneighbors(X_nn.loc[[idx]])
        similar = df.iloc[indices[0][1:]]
        st.dataframe(similar[['price','bedrooms','bathrooms','sqft_living','grade','zipcode']].round(0).reset_index(drop=True))

elif problem == "9. Waterfront Premium":
    st.header("Waterfront Premium")
    if 'waterfront' in df.columns and 'price' in df.columns and len(features) > 0:
        if df['waterfront'].sum() > 0:
            sample = df[df['waterfront'] == 1].iloc[[0]].copy()
            # ensure sample has all features
            sample_input = sample[features].select_dtypes(include=[np.number]).fillna(0)
            wf_price = safe_predict(model, sample_input)
            sample2 = sample.copy()
            sample2['waterfront'] = 0
            non_wf_price = safe_predict(model, sample2[features].select_dtypes(include=[np.number]).fillna(0))
            premium = wf_price - non_wf_price
            st.success(f"**Waterfront adds ≈ ${premium:,.0f}** (all else equal)")
        else:
            st.info("No waterfront properties found.")
    else:
        st.info("Required columns for this analysis are missing.")

elif problem == "10. Condition Segments":
    st.header("Price Segments by Condition")
    if {'condition','price'}.issubset(df.columns):
        Xc = df[['condition', 'price']].select_dtypes(include=[np.number]).fillna(0)
        df['segment'] = KMeans(n_clusters=3, random_state=42).fit_predict(Xc)
        fig, ax = plt.subplots()
        sns.boxplot(x='segment', y='price', data=df, palette="Set3", ax=ax)
        ax.set_title("Three Market Segments by Condition & Price")
        st.pyplot(fig)
    else:
        st.info("Required columns (condition, price) missing.")

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.caption("Built with ❤️ in Nigeria using Streamlit • Data: King County House Sales • November 17, 2025")
if not GROK_API_KEY:
    st.info("Want live AI explanations? Paste your Grok API key at the top!")
