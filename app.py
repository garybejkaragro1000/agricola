import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Soya: ARIMA/SARIMA/SARIMAX", layout="wide")

# ----------------------------
# Helpers
# ----------------------------

def try_parse_dates(s: pd.Series) -> pd.Series:
    # Acepta dd-mm-yyyy y dd/mm/yyyy; dayfirst=True para tu CSV
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def find_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in ["fecha", "date", "period", "time", "month", "fecha_mes"]]
    for c in df.columns:
        if c not in candidates:
            try:
                parsed = try_parse_dates(df[c])
                if parsed.notna().mean() > 0.8:
                    candidates.append(c)
            except Exception:
                pass
    return candidates[0] if len(candidates) else None

def ensure_monthly(series: pd.Series) -> pd.Series:
    s = series.dropna().sort_index()
    # Si no es mensual, agrego por fin de mes (mean)
    if s.index.inferred_freq in [None, "D", "B", "W", "QS", "Q", "A", "H", "T", "S"]:
        return s.resample("M").mean().dropna()
    # Si es mensual pero desalineado, fuerzo frecuencia M
    if s.index.inferred_freq and "M" in s.index.inferred_freq:
        return s.asfreq("M")
    return s.resample("M").mean().dropna()

def train_test_split_monthly(s: pd.Series, eval_start="2023-01-01", eval_end="2025-05-31") -> Tuple[pd.Series, pd.Series]:
    s = ensure_monthly(s)
    train = s[s.index < pd.to_datetime(eval_start)]
    test  = s[(s.index >= pd.to_datetime(eval_start)) & (s.index <= pd.to_datetime(eval_end))]
    return train, test

def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def fit_sarimax(y, order, seasonal_order=(0,0,0,0), exog=None):
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                    exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def ljungbox_pvalue(residuals, lags=24):
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return float(lb["lb_pvalue"].iloc[0])

def arch_pvalue(residuals, lags=12):
    stat, p, _, _ = het_arch(residuals, nlags=lags)
    return float(p)

def jb_pvalue(residuals):
    jb_stat, jb_p, _, _ = jarque_bera(residuals)
    return float(jb_p)

def fourier_terms(index, period=12, K=1):
    t = np.arange(len(index))
    X = {}
    for k in range(1, K+1):
        X[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        X[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(X, index=index)

def select_differencing(y: pd.Series) -> int:
    # ADF: si no estacionaria, d=1 (cap en 1)
    adf_p = adfuller(y.dropna(), autolag="AIC")[1]
    return 1 if adf_p > 0.05 else 0

def diagnostics(res, y, exog=None) -> Dict[str, float]:
    resid = res.resid.dropna()
    return {
        "jb_p": jb_pvalue(resid),
        "lb_p": ljungbox_pvalue(resid, lags=min(24, max(2, int(np.sqrt(len(resid)))))),
        "arch_p": arch_pvalue(resid, lags=min(12, max(2, int(np.sqrt(len(resid))/2))))),
        "resid": resid
    }

def record_result(kind, order, seasonal_order, exog_desc, test, fc, diag, aic):
    return {
        "model": kind,
        "order": order,
        "seasonal_order": seasonal_order,
        "exog": exog_desc,
        "aic": aic,
        "mape": mape(test.values, fc.values),
        "jb_p": diag["jb_p"],
        "lb_p": diag["lb_p"],
        "arch_p": diag["arch_p"],
        "forecast": fc
    }

def grid_search_models(train: pd.Series, test: pd.Series, seasonal_period=12, max_pq=3, K_fourier=(1,2,3)) -> Dict[str, Any]:
    results = []
    d = select_differencing(train)

    # ARIMA
    for p in range(0, max_pq+1):
        for q in range(0, max_pq+1):
            try:
                res = fit_sarimax(train, order=(p,d,q))
                fc = res.get_forecast(steps=len(test)).predicted_mean
                fc.index = test.index
                diag = diagnostics(res, train)
                results.append(record_result("ARIMA", (p,d,q), (0,0,0,0), None, test, fc, diag, res.aic))
            except Exception:
                continue

    # SARIMA
    for p in range(0, max_pq+1):
        for q in range(0, max_pq+1):
            for D in [0,1]:
                try:
                    res = fit_sarimax(train, order=(p,d,q), seasonal_order=(p, D, q, seasonal_period))
                    fc = res.get_forecast(steps=len(test)).predicted_mean
                    fc.index = test.index
                    diag = diagnostics(res, train)
                    results.append(record_result("SARIMA", (p,d,q), (p,D,q,seasonal_period), None, test, fc, diag, res.aic))
                except Exception:
                    continue

    # SARIMAX con términos de Fourier
    for K in K_fourier:
        exog_train = fourier_terms(train.index, period=seasonal_period, K=K)
        exog_test  = fourier_terms(test.index,  period=seasonal_period, K=K)
        for p in range(0, max_pq+1):
            for q in range(0, max_pq+1):
                for D in [0,1]:
                    try:
                        res = fit_sarimax(train, order=(p,d,q),
                                          seasonal_order=(p, D, q, seasonal_period),
                                          exog=exog_train)
                        fc = res.get_forecast(steps=len(test), exog=exog_test).predicted_mean
                        fc.index = test.index
                        diag = diagnostics(res, train, exog_train)
                        results.append(record_result(f"SARIMAX(K={K})", (p,d,q),
                                                     (p,D,q,seasonal_period),
                                                     f"Fourier K={K}", test, fc, diag, res.aic))
                    except Exception:
                        continue

    df = pd.DataFrame(results)
    if len(df)==0:
        return {"summary": pd.DataFrame(), "best": None}

    df["passes_all"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).all(axis=1)
    df["passes_count"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).sum(axis=1)

    candidates = df[df["passes_all"]].copy()
    if len(candidates) > 0:
        best_row = candidates.sort_values(["mape", "aic"]).iloc[0].to_dict()
    else:
        best_row = df.sort_values(["passes_count", "mape", "aic"],
                                  ascending=[False, True, True]).iloc[0].to_dict()
    return {"summary": df.sort_values(["passes_all","passes_count","mape","aic"],
                                      ascending=[False, False, True, True]),
            "best": best_row}

def plot_series(train, test, fc, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test (holdout)")
    fc.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def plot_residuals(resid, title_prefix=""):
    # Residual time plot
    fig, ax = plt.subplots(figsize=(10,3))
    resid.plot(ax=ax)
    ax.set_title(f"{title_prefix} Residuals over time")
    st.pyplot(fig)

    # Histograma
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(resid, bins=20, alpha=0.7)
    ax.set_title(f"{title_prefix} Residuals Histogram")
    st.pyplot(fig)

    # QQ
    fig = plt.figure(figsize=(6,3))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title(f"{title_prefix} Q-Q plot")
    st.pyplot(fig)

    # ACF / PACF
    fig_acf = plt.figure(figsize=(6,3))
    plot_acf(resid, lags=min(24, len(resid)//2), ax=plt.gca())
    plt.title(f"{title_prefix} Residuals ACF")
    st.pyplot(fig_acf)

    fig_pacf = plt.figure(figsize=(6,3))
    plot_pacf(resid, lags=min(24, len(resid)//2), ax=plt.gca(), method="ywm")
    plt.title(f"{title_prefix} Residuals PACF")
    st.pyplot(fig_pacf)

# ----------------------------
# UI principal
# ----------------------------

st.title("Modelado del Grano de Soya: ARIMA vs SARIMA vs SARIMAX")
st.caption("Criterios: normalidad, no autocorrelación, no heterocedasticidad y MAPE mínimo (enero 2023 a mayo 2025).")

with st.sidebar:
    st.header("Datos")
    default_path = "/mnt/data/soya_limpio_ddmmyyyy.csv"
    uploaded = st.file_uploader("Sube el CSV", type=["csv"])
    path = st.text_input("o ruta del CSV", value=default_path)
    eval_start = st.text_input("Inicio evaluación (YYYY-MM-DD)", value="2023-01-01")
    eval_end   = st.text_input("Fin evaluación (YYYY-MM-DD)", value="2025-05-31")
    max_pq = st.slider("Máx p y q", 1, 5, 3)
    seasonal_period = st.number_input("Periodo estacional (m)", min_value=4, max_value=24, value=12)
    K_min, K_max = st.slider("Fourier K (SARIMAX)", 1, 6, (1,3))

# Carga de datos
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if not Path(path).exists():
        st.error(f"No se encontró el archivo: {path}")
        st.stop()
    df = pd.read_csv(path)

st.subheader("Vista previa de datos")
st.dataframe(df.head(10))

date_col = find_date_col(df)
if date_col is None:
    st.error("No se detectó columna de fecha. Usa nombres como 'fecha' o 'date' o selecciona una existente.")
    st.stop()

df[date_col] = try_parse_dates(df[date_col])
df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

# Detectar objetivo (primera numérica)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.error("No se encontró una columna numérica para modelar (precio objetivo).")
    st.stop()
target = num_cols[0]
st.write(f"**Fecha:** `{date_col}` | **Objetivo:** `{target}`")

series = df[target].astype(float)
train, test = train_test_split_monthly(series, eval_start=eval_start, eval_end=eval_end)
st.write(f"Observaciones: Train={len(train)}, Test={len(test)}")
if len(train) < 36 or len(test) < 3:
    st.warning("Train o test es corto para una evaluación robusta. Verifica el rango de fechas.")

with st.expander("Descomposición STL (serie mensual)", expanded=False):
    s_monthly = ensure_monthly(series)
    stl = STL(s_monthly, period=int(seasonal_period), robust=True).fit()
    fig, axs = plt.subplots(4,1, figsize=(10,8), sharex=True)
    axs[0].plot(stl.observed); axs[0].set_title("Observado")
    axs[1].plot(stl.trend); axs[1].set_title("Tendencia")
    axs[2].plot(stl.seasonal); axs[2].set_title("Estacional")
    axs[3].plot(stl.resid); axs[3].set_title("Residuo")
    st.pyplot(fig)

st.subheader("Búsqueda y selección de modelos")
with st.spinner("Entrenando ARIMA / SARIMA / SARIMAX..."):
    out = grid_search_models(train, test, seasonal_period=int(seasonal_period),
                             max_pq=int(max_pq), K_fourier=range(K_min, K_max+1))

if out["best"] is None or len(out["summary"]) == 0:
    st.error("No fue posible ajustar modelos válidos. Revisa los datos.")
    st.stop()

st.markdown("### Ranking de modelos")
display_cols = ["model", "order", "seasonal_order", "exog", "aic", "mape", "jb_p", "lb_p", "arch_p"]
st.dataframe(out["summary"][display_cols].style.format({
    "aic":"{:.1f}", "mape":"{:.2f}%", "jb_p":"{:.3f}", "lb_p":"{:.3f}", "arch_p":"{:.3f}"
}))

best = out["best"]
st.success(f"**Mejor modelo:** {best['model']} | order={best['order']} | seasonal={best['seasonal_order']} "
           f"| exog={best['exog']} | MAPE={best['mape']:.2f}%")

# Recuperar el forecast de la fila ganadora
def find_forecast_series(summary_df, best_dict):
    mask = (summary_df["model"]==best_dict["model"]) & \
           (summary_df["order"]==tuple(best_dict["order"])) & \
           (summary_df["seasonal_order"]==tuple(best_dict["seasonal_order"])) & \
           (summary_df["mape"]==best_dict["mape"])
    row = summary_df[mask].iloc[0]
    return row["forecast"]

fc = find_forecast_series(out["summary"], best)

st.markdown("### Pronóstico vs. Real (ventana de evaluación)")
plot_series(train, test, fc, "Pronóstico sobre evaluación (2023-01 a 2025-05)")

st.markdown("### Diagnósticos del mejor modelo (p-values)")
st.write({
    "Jarque–Bera (normalidad)": round(float(best["jb_p"]), 4),
    "Ljung–Box (no autocorrelación)": round(float(best["lb_p"]), 4),
    "ARCH (no heterocedasticidad)": round(float(best["arch_p"]), 4),
})

# Reentreno sólo para visualizar residuos de train del mejor
def refit_and_get_resid(model_name, order, seasonal_order, train, seasonal_period):
    if model_name.startswith("SARIMAX"):
        import re
        k = int(re.search(r"K=(\d+)", model_name).group(1))
        exog_train = fourier_terms(train.index, period=seasonal_period, K=k)
        res = SARIMAX(train, order=order, seasonal_order=seasonal_order, exog=exog_train,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    else:
        res = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return res.resid.dropna()

resid_best = refit_and_get_resid(best["model"], tuple(best["order"]), tuple(best["seasonal_order"]),
                                 train, int(seasonal_period))
plot_residuals(resid_best, title_prefix=f"{best['model']}")

# Exportables
st.markdown("### Exportar resultados")
export = {
    "best_model": {
        "model": best["model"],
        "order": tuple(best["order"]),
        "seasonal_order": tuple(best["seasonal_order"]),
        "exog": best["exog"],
        "aic": float(best["aic"]),
        "mape_eval_%": float(best["mape"]),
        "diagnostics_pvalues": {
            "jarque_bera": float(best["jb_p"]),
            "ljung_box": float(best["lb_p"]),
            "arch": float(best["arch_p"])
        }
    },
    "evaluation_window": {
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max())
    },
}

json_bytes = io.BytesIO()
json_bytes.write(pd.Series(export).to_json().encode("utf-8"))
json_bytes.seek(0)
st.download_button("Descargar resumen JSON", data=json_bytes,
                   file_name="soya_model_summary.json", mime="application/json")

csv_bytes = io.BytesIO()
pd.DataFrame({"real": test, "forecast": fc}).to_csv(csv_bytes, index=True)
csv_bytes.seek(0)
st.download_button("Descargar pronóstico (CSV)", data=csv_bytes,
                   file_name="forecast_eval.csv", mime="text/csv")
