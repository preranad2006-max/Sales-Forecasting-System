# Sales Forecasting System

An end-to-end sales forecasting project for college coursework and viva demonstrations. It cleans historical sales data, surfaces demand patterns, compares machine learning and time series approaches, and produces daily, weekly, and monthly planning forecasts.

## What is included

- A polished browser dashboard in `artifacts/sales-forecasting-system`
- A Python/Streamlit implementation in `streamlit_app.py`
- A realistic sample dataset in `data/sample_sales_data.csv`
- CSV and Excel upload support in the Streamlit app and browser dashboard
- Validation, duplicate removal, invalid-row handling, trend and seasonality analysis
- Linear Regression, Random Forest Regressor, ARIMA, and seasonal-naive baselines
- MAE, MSE, and RMSE model evaluation on a chronological holdout set
- Daily, weekly, and monthly forecast views with uncertainty bands
- CSV exports, cleaned-data export, and a viva-ready methodology summary

## Run the browser dashboard

```bash
pnpm install
pnpm --filter @workspace/sales-forecasting-system run dev
```

The dashboard is designed to run inside the Replit preview. Uploads are processed locally in the browser, so the sample experience works without an API key or external service.

## Run the Streamlit version

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dataset schema

The required columns are:

| Column | Type | Meaning |
| --- | --- | --- |
| `Date` | date | Observation date |
| `Product_ID` | text | Product identifier |
| `Store_ID` | text | Store identifier |
| `Units_Sold` | number | Demand target |
| `Revenue` | number | Sales revenue |
| `Promotion` | boolean | Whether a promotion was active |
| `Holiday` | boolean | Whether the date was a holiday |

## Modeling approach

1. Parse dates and numeric fields, normalize boolean flags, clip negative values, remove duplicates, and discard invalid records.
2. Aggregate demand to a daily time series and derive lag, trend, weekday, month, promotion, and holiday features.
3. Split chronologically so future observations never leak into training.
4. Compare Linear Regression, Random Forest Regressor, ARIMA, and seasonal-naive reference forecasts.
5. Select the lowest RMSE model and expose MAE, MSE, RMSE, confidence bands, and planning recommendations.

## Viva talking points

- Time-based splits are safer than random splits for forecasting because the model must predict future periods.
- MAE is easy to interpret in units, while RMSE penalizes large mistakes more heavily.
- Random Forest can capture non-linear promotion and product/store interactions; ARIMA is valuable when the series’ own temporal structure dominates.
- A forecast is a planning input, not a guarantee, so the dashboard shows an uncertainty band and an operating action.