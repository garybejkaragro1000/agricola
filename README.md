# Soya Time-Series Modeling App (ARIMA / SARIMA / SARIMAX)

This Streamlit app models the soybean (soya) price series from the attached dataset and evaluates ARIMA,
SARIMA, and SARIMAX. The best model must pass diagnostics (normality, no autocorrelation, no heteroskedasticity)
and have the lowest MAPE on the evaluation window Jan 2023 – May 2025.

## How to run

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Default path in the app: `/mnt/data/soya_limpio_ddmmyyyy.csv`.
You can also upload a CSV or change the path in the sidebar.

- The app auto-detects the date column (`fecha`, `date`, `period`, etc.) and the first numeric target column.
- Dates are parsed with `dayfirst=True`. If the data is daily, it's aggregated to monthly means.
