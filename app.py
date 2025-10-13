# ======================================================
# app.py actualizado con bloque de pruebas de normalidad estilo EViews
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Soya ARIMA/SARIMA/SARIMAX Selector", layout="wide")

# ============================
# Funciones auxiliares (índice, limpieza, modelado)
# ============================

def to_month_end_index(idx) -> pd.DatetimeIndex:
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp('M')
    idx = pd.to_datetime(idx, errors="coerce")
    return idx.to_period('M').to_timestamp('M')

def replace_nonfinite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def winsorize_series(s: pd.Series, low_q=0.01, high_q=0.99):
    lo, hi = s.quantile(low_q), s.quantile(high_q)
    return s.clip(lower=lo, upper=hi), (s < lo) | (s > hi)

def interpolate_series(s: pd.Series, method="linear"):
    before = s.isna()
    s2 = s.interpolate(method=method, limit_direction="both")
    return s2, before & (~s2.isna())

def apply_log(s: pd.Series, mode="none"):
    if mode == "none": return s
    if mode == "log": return np.log(s.where(s > 0))
    if mode == "log1p": return np.log1p(s.where(s >= -0.999999))
    return s

def clean_pipeline(s: pd.Series, do_winsor=True, q_low=0.01, q_high=0.99,
                   do_interp=True, interp_method="linear",
                   do_ffill=True, do_bfill=True, log_mode="none"):
    s0 = replace_nonfinite(s)
    if do_winsor:
        s1, _ = winsorize_series(s0, q_low, q_high)
    else:
        s1 = s0.copy()
    if do_interp:
        s2, _ = interpolate_series(s1, interp_method)
    else:
        s2 = s1.copy()
    if do_ffill: s2 = s2.ffill()
    if do_bfill: s2 = s2.bfill()
    s3 = apply_log(s2, log_mode)
    return s3, {}, {}

def ensure_monthly_series(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    try:
        s_m = s.resample("M").mean().dropna()
    except Exception:
        s_m = s
    s_m.index = to_month_end_index(s_m.index)
    return s_m

def train_test_split_monthly(s: pd.Series, eval_start="2023-01-01", eval_end="2025-05-31") -> Tuple[pd.Series, pd.Series]:
    s = ensure_monthly_series(s)
    train = s[s.index < pd.to_datetime(eval_start)]
    test  = s[(s.index >= pd.to_datetime(eval_start)) & (s.index <= pd.to_datetime(eval_end))]
    train.index = to_month_end_index(train.index)
    test.index  = to_month_end_index(test.index)
    return train, test

def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def fit_sarimax(y, order, seasonal_order=(0,0,0,0), exog=None):
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

def diagnostics(res) -> Dict[str, float]:
    resid = res.resid.dropna()
    lb_lags = min(24, max(2, int(np.sqrt(len(resid)))))
    arch_lags = min(12, max(2, int(np.sqrt(len(resid)) // 2)))
    
    jb_stat, jb_p = stats.jarque_bera(resid_best)

    lb_p = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)["lb_pvalue"].iloc[0]
    arch_p = het_arch(resid, nlags=arch_lags)[1]
    return {"jb_p": jb_p, "lb_p": lb_p, "arch_p": arch_p, "resid": resid}

# ============================
# Interfaz Streamlit
# ============================

st.title("Modelado del Grano de Soya: ARIMA vs SARIMA vs SARIMAX")

# Datos
uploaded = st.file_uploader("Sube el CSV de precios de soya", type=["csv"])
if uploaded is None:
    st.stop()

st.subheader("Vista previa de datos")
df = pd.read_csv(uploaded)
st.dataframe(df.head())

fecha_col = [c for c in df.columns if 'fecha' in c.lower() or 'date' in c.lower()][0]
precio_col = [c for c in df.columns if c != fecha_col][0]
df[fecha_col] = pd.to_datetime(df[fecha_col], dayfirst=True)
df = df.set_index(fecha_col).sort_index()

series, _, _ = clean_pipeline(df[precio_col])
train, test = train_test_split_monthly(series)

st.write(f"Train: {len(train)} observaciones, Test: {len(test)}")

# Modelo SARIMAX simple de ejemplo
res = fit_sarimax(train, order=(1,1,1), seasonal_order=(1,0,1,12))
fc = res.get_forecast(steps=len(test)).predicted_mean

# Resultados
st.line_chart(pd.DataFrame({"Real": test, "Pronóstico": fc}))

# === Diagnósticos del modelo ===
resid_best = res.resid.dropna()

st.markdown("### Diagnósticos del mejor modelo")
st.write({
    "Jarque–Bera p-value (normalidad)": round(float(diagnostics(res)["jb_p"]), 4),
    "Ljung–Box p-value (no autocorrelación)": round(float(diagnostics(res)["lb_p"]), 4),
    "ARCH p-value (no heterocedasticidad)": round(float(diagnostics(res)["arch_p"]), 4),
})

# === Pruebas de normalidad estilo EViews ===
st.markdown("### Pruebas de Normalidad de los Residuales (estilo EViews)")

resid_best = resid_best.dropna()
mean_resid = np.mean(resid_best)
std_resid = np.std(resid_best, ddof=1)
skew_resid = stats.skew(resid_best)
kurt_resid = stats.kurtosis(resid_best, fisher=False)

jb_stat, jb_p = stats.jarque_bera(resid_best)


df_norm = pd.DataFrame({
    "Estadístico": [
        "Media de los residuales",
        "Desviación estándar",
        "Asimetría (Skewness)",
        "Curtosis (Kurtosis)",
        "Jarque–Bera",
        "Probabilidad (JB)"
    ],
    "Valor": [
        f"{mean_resid:.6f}",
        f"{std_resid:.6f}",
        f"{skew_resid:.6f}",
        f"{kurt_resid:.6f}",
        f"{jb_stat:.6f}",
        f"{jb_p:.6f}"
    ]
})
st.table(df_norm)

if jb_p > 0.05:
    st.success("✅ Los residuales siguen una distribución normal (no se rechaza H₀ de normalidad).")
else:
    st.warning("⚠️ Los residuales **no** siguen una distribución normal (se rechaza H₀ de normalidad).")

# Histograma + curva normal
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(resid_best, bins=20, color="skyblue", edgecolor="black", alpha=0.7, density=True)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean_resid, std_resid)
ax.plot(x, p, 'r', linewidth=2)
ax.set_title("Histograma de los Residuales con Curva Normal (EViews Style)")
ax.set_xlabel("Residuales"); ax.set_ylabel("Densidad")
st.pyplot(fig)

# Q–Q Plot
fig = plt.figure(figsize=(6,4))
stats.probplot(resid_best, dist="norm", plot=plt)
plt.title("Q–Q Plot de los Residuales (EViews Style)")
st.pyplot(fig)
