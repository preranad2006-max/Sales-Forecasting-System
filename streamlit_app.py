"""Sales Forecasting System

Run with:
    streamlit run streamlit_app.py

The browser dashboard in artifacts/sales-forecasting-system is the polished
interactive preview. This Streamlit app is the accompanying Python deliverable
for local use, coursework, and viva demonstrations.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

REQUIRED_COLUMNS = ["Date", "Product_ID", "Store_ID", "Units_Sold", "Revenue", "Promotion", "Holiday"]
SAMPLE_PATH = Path(__file__).parent / "data" / "sample_sales_data.csv"


def parse_flag(value: object) -> int:
    return int(str(value).strip().lower() in {"true", "1", "yes", "y", "on"})


@st.cache_data
def load_data(file_bytes: bytes | None, filename: str | None) -> tuple[pd.DataFrame, dict]:
    if file_bytes is None:
        raw = pd.read_csv(SAMPLE_PATH)
        source = "Bundled sample dataset"
    elif filename and filename.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(BytesIO(file_bytes))
        source = filename
    else:
        raw = pd.read_csv(BytesIO(file_bytes))
        source = filename or "Uploaded CSV"

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    clean = raw[REQUIRED_COLUMNS].copy()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean["Units_Sold"] = pd.to_numeric(clean["Units_Sold"], errors="coerce")
    clean["Revenue"] = pd.to_numeric(clean["Revenue"], errors="coerce")
    clean["Promotion"] = clean["Promotion"].map(parse_flag)
    clean["Holiday"] = clean["Holiday"].map(parse_flag)
    before = len(clean)
    invalid = clean[["Date", "Product_ID", "Store_ID", "Units_Sold", "Revenue"]].isna().any(axis=1)
    clean = clean.loc[~invalid].copy()
    clean["Units_Sold"] = clean["Units_Sold"].clip(lower=0)
    clean["Revenue"] = clean["Revenue"].clip(lower=0)
    clean = clean.drop_duplicates(subset=["Date", "Product_ID", "Store_ID"])
    clean = clean.sort_values("Date").reset_index(drop=True)
    metadata = {
        "source": source,
        "rows_before": before,
        "rows_after": len(clean),
        "invalid": int(invalid.sum()),
        "duplicates": before - int(invalid.sum()) - len(clean),
    }
    return clean, metadata


def daily_series(data: pd.DataFrame) -> pd.Series:
    return data.set_index("Date")["Units_Sold"].resample("D").sum().asfreq("D", fill_value=0)


def make_features(series: pd.Series, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(index=series.index)
    frame["trend"] = np.arange(len(frame))
    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    for lag in (1, 7, 14, 30):
        frame[f"lag_{lag}"] = series.shift(lag)
    commercial = data.groupby("Date")[["Promotion", "Holiday"]].max().reindex(frame.index).fillna(0)
    frame = frame.join(commercial)
    frame = frame.dropna()
    return frame, series.loc[frame.index]


def train_models(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    series = daily_series(data)
    features, target = make_features(series, data)
    holdout = max(7, int(len(features) * 0.2))
    x_train, x_test = features.iloc[:-holdout], features.iloc[-holdout:]
    y_train, y_test = target.iloc[:-holdout], target.iloc[-holdout:]
    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=160, random_state=42, max_depth=10),
    }
    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mean_squared_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        })
        fitted[name] = model

    naive_predictions = y_test.shift(7).fillna(y_train.mean())
    results.append({
        "Model": "Seasonal naive",
        "MAE": mean_absolute_error(y_test, naive_predictions),
        "MSE": mean_squared_error(y_test, naive_predictions),
        "RMSE": mean_squared_error(y_test, naive_predictions) ** 0.5,
    })

    arima_order = (1, 1, 1)
    try:
        arima = ARIMA(series.iloc[:-holdout], order=arima_order).fit()
        arima_predictions = arima.forecast(holdout)
        results.append({
            "Model": "ARIMA (1,1,1)",
            "MAE": mean_absolute_error(series.iloc[-holdout:], arima_predictions),
            "MSE": mean_squared_error(series.iloc[-holdout:], arima_predictions),
            "RMSE": mean_squared_error(series.iloc[-holdout:], arima_predictions) ** 0.5,
        })
        fitted["ARIMA (1,1,1)"] = arima
    except Exception:
        st.info("ARIMA could not converge for this upload; the other models are still available.")

    metrics = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return metrics, {"series": series, "features": features, "target": target, "fitted": fitted, "holdout": holdout}


def forecast_future(data: pd.DataFrame, model_name: str, horizon: int, fitted: dict[str, object]) -> pd.DataFrame:
    series = daily_series(data)
    if model_name == "ARIMA (1,1,1)" and model_name in fitted:
        values = fitted[model_name].forecast(horizon)
    elif model_name in fitted:
        model = fitted[model_name]
        history = series.copy()
        points = []
        for _ in range(horizon):
            date = history.index[-1] + pd.Timedelta(days=1)
            features = pd.DataFrame({
                "trend": [len(history)],
                "day_of_week": [date.dayofweek],
                "month": [date.month],
                "lag_1": [history.iloc[-1]],
                "lag_7": [history.iloc[-7] if len(history) >= 7 else history.mean()],
                "lag_14": [history.iloc[-14] if len(history) >= 14 else history.mean()],
                "lag_30": [history.iloc[-30] if len(history) >= 30 else history.mean()],
                "Promotion": [0],
                "Holiday": [0],
            })
            prediction = max(0, float(model.predict(features)[0]))
            points.append(prediction)
            history.loc[date] = prediction
        values = points
    else:
        values = [series.tail(14).mean()] * horizon
    result = pd.DataFrame({"Date": pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon), "Forecast": values})
    result["Lower"] = result["Forecast"] * 0.9
    result["Upper"] = result["Forecast"] * 1.1
    return result


def show_analysis(data: pd.DataFrame) -> None:
    st.subheader("Exploratory data analysis")
    left, right = st.columns(2)
    with left:
        st.line_chart(data.set_index("Date")["Units_Sold"].resample("W").sum(), height=280)
        st.caption("Weekly units sold")
    with right:
        monthly = data.set_index("Date")["Units_Sold"].resample("MS").sum()
        st.bar_chart(monthly, height=280)
        st.caption("Monthly seasonality")
    summary = data.groupby("Promotion")["Units_Sold"].mean().rename({0: "Regular", 1: "Promotion"}).to_frame("Average units")
    st.dataframe(summary.style.format("{:.1f}"), use_container_width=True)


st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")
st.title("Sales Forecasting System")
st.caption("Machine learning and time series planning for daily, weekly, and monthly demand.")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
page = st.sidebar.radio("Workflow", ["Dataset Upload", "Data Analysis", "Model Training", "Sales Forecasting", "Results"])

try:
    data, metadata = load_data(uploaded.getvalue() if uploaded else None, uploaded.name if uploaded else None)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.success(f"{len(data):,} clean rows ready")
st.sidebar.caption(f"Source: {metadata['source']}")

if page == "Dataset Upload":
    st.header("Dataset upload and validation")
    st.write("Required columns:", ", ".join(REQUIRED_COLUMNS))
    st.dataframe(data.head(20), use_container_width=True)
    cards = st.columns(4)
    cards[0].metric("Rows inspected", f"{metadata['rows_before']:,}")
    cards[1].metric("Valid rows", f"{metadata['rows_after']:,}")
    cards[2].metric("Duplicates removed", metadata["duplicates"])
    cards[3].metric("Rows flagged", metadata["invalid"])
    st.download_button("Download cleaned CSV", data.to_csv(index=False), "cleaned_sales_data.csv", "text/csv")
elif page == "Data Analysis":
    show_analysis(data)
elif page == "Model Training":
    st.header("Model comparison")
    metrics, artifacts = train_models(data)
    st.dataframe(metrics.style.format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}"}), use_container_width=True)
    best = metrics.iloc[0]
    st.success(f"Best-performing model: {best['Model']} with RMSE {best['RMSE']:.2f}")
elif page == "Sales Forecasting":
    st.header("Sales forecasting")
    metrics, artifacts = train_models(data)
    model_name = st.selectbox("Model", metrics["Model"].tolist())
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
    result = forecast_future(data, model_name, horizon, artifacts["fitted"])
    st.line_chart(result.set_index("Date")[["Forecast", "Lower", "Upper"]], height=360)
    st.dataframe(result, use_container_width=True)
    st.download_button("Download forecast", result.to_csv(index=False), "sales_forecast.csv", "text/csv")
else:
    st.header("Results and viva summary")
    metrics, artifacts = train_models(data)
    best = metrics.iloc[0]
    st.metric("Selected model", best["Model"])
    st.metric("Validation RMSE", f"{best['RMSE']:.2f}")
    st.markdown(
        "**Methodology:** clean the seven required fields, inspect trends and seasonality, "
        "split the time series chronologically, train multiple approaches, compare MAE/MSE/RMSE, "
        "and use the lowest-error model to create an uncertainty-aware forecast."
    )
    st.dataframe(metrics, use_container_width=True)