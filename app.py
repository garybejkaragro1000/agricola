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

st.set_page_config(page_title="Soya ARIMA/SARIMA/SARIMAX Selector", layout="wide")

# ----------------------------
# Limpieza de datos
# ----------------------------

def replace_nonfinite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def winsorize_series(s: pd.Series, low_q=0.01, high_q=0.99):
    s_clean = s.copy()
    lo = s_clean.quantile(low_q)
    hi = s_clean.quantile(high_q)
    capped = (s_clean < lo) | (s_clean > hi)
    return s_clean.clip(lower=lo, upper=hi), capped

def interpolate_series(s: pd.Series, method="linear"):
    before = s.isna()
    s2 = s.interpolate(method=method, limit_direction="both")
    after = s2.isna()
    filled = before & (~after)
    return s2, filled

def apply_log(s: pd.Series, mode="none"):
    if mode == "none":
        return s
    if mode == "log":
        s_pos = s.where(s > 0, np.nan)
        return np.log(s_pos)
    if mode == "log1p":
        s_shift = s.where(s >= -0.999999, np.nan)
        return np.log1p(s_shift)
    return s

def clean_pipeline(s: pd.Series, do_winsor=True, q_low=0.01, q_high=0.99,
                   do_interp=True, interp_method="linear",
                   do_ffill=True, do_bfill=True,
                   log_mode="none"):
    report = {}

    s0 = replace_nonfinite(s)
    report["initial_na"] = int(s0.isna().sum())

    if do_winsor:
        s1, capped_mask = winsorize_series(s0, q_low, q_high)
    else:
        s1, capped_mask = s0.copy(), pd.Series(False, index=s0.index)
    report["winsor_applied"] = bool(do_winsor)

    if do_interp:
        s2, interp_mask = interpolate_series(s1, interp_method)
    else:
        s2, interp_mask = s1.copy(), pd.Series(False, index=s1.index)

    if do_ffill:
        pre_ffill_na = s2.isna()
        s2 = s2.ffill()
        ffill_mask = pre_ffill_na & (~s2.isna())
    else:
        ffill_mask = pd.Series(False, index=s2.index)

    if do_bfill:
        pre_bfill_na = s2.isna()
        s2 = s2.bfill()
        bfill_mask = pre_bfill_na & (~s2.isna())
    else:
        bfill_mask = pd.Series(False, index=s2.index)

    report["after_fill_na"] = int(s2.isna().sum())

    s3 = apply_log(s2, log_mode)
    report["transform"] = log_mode
    report["final_na"] = int(s3.isna().sum())
    report["length"] = int(len(s3))

    masks = {"capped": capped_mask, "interp": interp_mask, "ffill": ffill_mask, "bfill": bfill_mask}
    return s3, report, masks

# ----------------------------
# Helpers de modelado
# ----------------------------

def try_parse_dates(s: pd.Series) -> pd.Series:
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
    if s.index.inferred_freq in [None, "D", "B", "W", "QS", "Q", "A", "H", "T", "S"]:
        return s.resample("M").mean().dropna()
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
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, exog=exog,
                    enforce_stationarity=False, enforce_invertibility=False)
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
    y_clean = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(y_clean) < 12:
        return 0
    if y_clean.std() == 0 or y_clean.max() == y_clean.min():
        return 0
    try:
        adf_p = adfuller(y_clean.values, autolag="AIC")[1]
        return 1 if adf_p > 0.05 else 0
    except Exception:
        return 0

def diagnostics(res) -> Dict[str, float]:
    resid = res.resid.dropna()
    lb_lags = min(24, max(2, int(np.sqrt(len(resid)))))
    arch_lags = min(12, max(2, int(np.sqrt(len(resid)) // 2)))
    return {
        "jb_p": jb_pvalue(resid),
        "lb_p": ljungbox_pvalue(resid, lags=lb_lags),
        "arch_p": arch_pvalue(resid, lags=arch_lags),
        "resid": resid
    }

def record_result(kind, order, seasonal_order, exog_desc, test, fc, diag, aic):
    return {
        "model": kind,
        "order": order,
        "seasonal_order": seasonal_order,
        "exog": exog_desc,
        "aic": aic,
        "mape": mape(test.values, fc.values) if (test is not None and fc is not None) else np.inf,
        "jb_p": diag["jb_p"],
        "lb_p": diag["lb_p"],
        "arch_p": diag["arch_p"],
        "forecast": fc
    }

def grid_search_models(train: pd.Series, test: pd.Series, seasonal_period=12, max_pq=3,
                       K_fourier=(1,2,3), mape_threshold=None, enforce_threshold=False) -> Dict[str, Any]:
    results = []
    d = select_differencing(train)

    # ARIMA
    for p in range(0, max_pq+1):
        for q in range(0, max_pq+1):
            try:
                res = fit_sarimax(train, order=(p,d,q))
                fc = res.get_forecast(steps=len(test)).predicted_mean
                fc.index = test.index
                diag = diagnostics(res)
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
                    diag = diagnostics(res)
                    results.append(record_result("SARIMA", (p,d,q), (p,D,q,seasonal_period), None, test, fc, diag, res.aic))
                except Exception:
                    continue

    # SARIMAX (Fourier exog)
    for K in K_fourier:
        exog_train = fourier_terms(train.index, period=seasonal_period, K=K)
        exog_test  = fourier_terms(test.index,  period=seasonal_period, K=K)
        for p in range(0, max_pq+1):
            for q in range(0, max_pq+1):
                for D in [0,1]:
                    try:
                        res = fit_sarimax(train, order=(p,d,q), seasonal_order=(p, D, q, seasonal_period), exog=exog_train)
                        fc = res.get_forecast(steps=len(test), exog=exog_test).predicted_mean
                        fc.index = test.index
                        diag = diagnostics(res)
                        results.append(record_result(f"SARIMAX(K={K})", (p,d,q), (p,D,q,seasonal_period), f"Fourier K={K}", test, fc, diag, res.aic))
                    except Exception:
                        continue

    df = pd.DataFrame(results)
    if len(df)==0:
        return {"summary": pd.DataFrame(), "best": None}

    df["passes_all"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).all(axis=1)
    df["passes_count"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).sum(axis=1)
    df["meets_thresh"] = df["mape"] <= (mape_threshold if mape_threshold is not None else np.inf)

    best_row = None

    if mape_threshold is not None:
        diag_and_thr = df[df["passes_all"] & df["meets_thresh"]]
        if len(diag_and_thr) > 0:
            best_row = diag_and_thr.sort_values(["mape","aic"]).iloc[0].to_dict()
        else:
            if enforce_threshold:
                diag_only = df[df["passes_all"]]
                if len(diag_only) > 0:
                    best_row = diag_only.sort_values(["mape","aic"]).iloc[0].to_dict()
                    best_row["_note"] = f"Ningún modelo cumple MAPE ≤ {mape_threshold:.2f}%. Se muestra el mejor que pasa diagnósticos."
                else:
                    thr_only = df[df["meets_thresh"]]
                    if len(thr_only) > 0:
                        best_row = thr_only.sort_values(["mape","aic"]).iloc[0].to_dict()
                        best_row["_note"] = "No hay modelo que pase todos los diagnósticos. Se muestra el de menor MAPE que cumple el umbral."
                    else:
                        best_row = df.sort_values(["passes_count","mape","aic"], ascending=[False, True, True]).iloc[0].to_dict()
                        best_row["_note"] = "No hay modelos que cumplan el umbral MAPE ni los diagnósticos completos. Mostrando el mejor compromiso."
            else:
                diag_only = df[df["passes_all"]]
                if len(diag_only) > 0:
                    best_row = diag_only.sort_values(["mape","aic"]).iloc[0].to_dict()
                else:
                    best_row = df.sort_values(["passes_count","mape","aic"], ascending=[False, True, True]).iloc[0].to_dict()
    else:
        diag_only = df[df["passes_all"]]
        if len(diag_only) > 0:
            best_row = diag_only.sort_values(["mape","aic"]).iloc[0].to_dict()
        else:
            best_row = df.sort_values(["passes_count","mape","aic"], ascending=[False, True, True]).iloc[0].to_dict()

    return {"summary": df.sort_values(["passes_all","passes_count","mape","aic"], ascending=[False, False, True, True]), "best": best_row}

def plot_series(train, test, fc, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test (holdout)")
    if fc is not None:
        fc.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def plot_residuals(resid, title_prefix=""):
    # Serie de residuos
    fig, ax = plt.subplots(figsize=(10,3))
    resid.plot(ax=ax)
    ax.set_title(f"{title_prefix} Residuals over time")
    st.pyplot(fig)

    # Histograma
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(resid, bins=20, alpha=0.7)
    ax.set_title(f"{title_prefix} Residuals Histogram")
    st.pyplot(fig)

    # Q-Q
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

def fit_best_model_for_summary(best: dict, train: pd.Series, seasonal_period: int):
    """Refit del mejor modelo sobre el conjunto de entrenamiento para obtener .summary()."""
    model_name = str(best.get("model", ""))
    order = tuple(best.get("order", (0, 0, 0)))
    seasonal_order = tuple(best.get("seasonal_order", (0, 0, 0, 0)))

    if model_name.startswith("SARIMAX"):
        import re as _re
        m = _re.search(r"K=(\d+)", model_name)
        k = int(m.group(1)) if m else 1
        exog_train = fourier_terms(train.index, period=int(seasonal_period), K=k)
        res = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    else:
        res = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return res

def find_forecast_series(summary_df, best_dict):
    """Encuentra la serie de pronóstico para el modelo 'best'. Si no coincide exactamente, usa la primera fila como respaldo."""
    try:
        mask = (
            (summary_df["model"] == best_dict.get("model")) &
            (summary_df["order"] == tuple(best_dict.get("order", (0,0,0)))) &
            (summary_df["seasonal_order"] == tuple(best_dict.get("seasonal_order", (0,0,0,0)))) &
            (summary_df["mape"] == best_dict.get("mape"))
        )
        subset = summary_df[mask]
        if len(subset) == 0 and len(summary_df) > 0:
            subset = summary_df.iloc[[0]]
        row = subset.iloc[0]
        return row.get("forecast", None)
    except Exception:
        return None

def refit_and_get_resid(model_name, order, seasonal_order, train, seasonal_period):
    if model_name.startswith("SARIMAX"):
        import re
        m = re.search(r"K=(\d+)", model_name)
        k = int(m.group(1)) if m else 1
        exog_train = fourier_terms(train.index, period=seasonal_period, K=k)
        res = SARIMAX(train, order=order, seasonal_order=seasonal_order, exog=exog_train,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    else:
        res = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return res.resid.dropna()

# ----------------------------
# UI
# ----------------------------

st.title("Modelado del Grano de Soya: ARIMA vs SARIMA vs SARIMAX")
st.caption("Criterios: normalidad, no autocorrelación, no heterocedasticidad y MAPE mínimo en la evaluación (2023-01 a 2025-05).")

with st.sidebar:
    st.header("Datos")
    default_path = "C:\\Tito\\Gary\\soya_limpio_ddmmyyyy.csv"
    #default_path = "/mnt/data/soya_limpio_ddmmyyyy.csv"
    uploaded = st.file_uploader("Sube el CSV", type=["csv"])
    path = st.text_input("o ruta del CSV", value=default_path)
    eval_start = st.text_input("Inicio evaluación (YYYY-MM-DD)", value="2023-01-01")
    eval_end   = st.text_input("Fin evaluación (YYYY-MM-DD)", value="2025-05-31")
    max_pq = st.slider("Máx p y q", 1, 5, 3)
    seasonal_period = st.number_input("Periodo estacional (m)", min_value=4, max_value=24, value=12)
    K_min, K_max = st.slider("Fourier K (SARIMAX)", 1, 6, (1,3))

    st.header("Limpieza de datos")
    do_winsor = st.checkbox("Capar outliers (winsorizar)", value=True)
    q_low, q_high = st.slider("Percentiles de capping", 0.0, 0.2, (0.01, 0.99))
    do_interp = st.checkbox("Interpolar huecos (lineal)", value=True)
    interp_method = st.selectbox("Método de interpolación", ["linear","time","nearest","polynomial"], index=0)
    do_ffill = st.checkbox("Relleno hacia adelante (ffill)", value=True)
    do_bfill = st.checkbox("Relleno hacia atrás (bfill)", value=True)
    log_mode = st.selectbox("Transformación", ["none","log","log1p"], index=0)

    robust_mode = st.checkbox("Modo robusto (auto-sanar)", value=True)
    min_train_months = st.number_input("Mínimo meses en Train", min_value=6, max_value=48, value=12)

    st.header("Criterio MAPE")
    mape_thr = st.number_input("Umbral MAPE objetivo (%)", min_value=0.1, max_value=50.0, value=4.0, step=0.1)
    enforce_thr = st.checkbox("Exigir umbral MAPE (descartar modelos que no lo cumplan)", value=True)

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
    st.error("No se pudo detectar la columna de fecha. Por favor, selecciona una columna válida llamada por ejemplo 'fecha' o 'date'.")
    st.stop()

df[date_col] = try_parse_dates(df[date_col])
df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

target_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not target_candidates:
    st.error("No se encontró una columna numérica para modelar (precio).")
    st.stop()
target = target_candidates[0]
st.write(f"**Fecha:** `{date_col}` | **Objetivo:** `{target}`")

series_raw = df[target].astype(float)
series, clean_report, masks = clean_pipeline(series_raw, do_winsor, q_low, q_high, do_interp, interp_method, do_ffill, do_bfill, log_mode)

with st.expander("Reporte de limpieza", expanded=False):
    st.write({
        "NA iniciales": clean_report["initial_na"],
        "Winsor aplicado": clean_report["winsor_applied"],
        "NA después de interpolar/ffill/bfill": clean_report["after_fill_na"],
        "Transformación": clean_report["transform"],
        "NA finales": clean_report["final_na"],
        "Observaciones totales": clean_report["length"]
    })
    fig, ax = plt.subplots(figsize=(10,3))
    ensure_monthly(series_raw).plot(ax=ax, label="Crudo (mensualizado)")
    ensure_monthly(series).plot(ax=ax, label="Limpio (mensualizado)")
    ax.set_title("Validación de limpieza")
    ax.legend()
    st.pyplot(fig)
    interventions = pd.DataFrame({
        "capped": masks["capped"],
        "interp": masks["interp"],
        "ffill": masks["ffill"],
        "bfill": masks["bfill"],
    })
    if interventions.any(axis=1).any():
        st.write("Matriz de intervenciones (True=aplicado):")
        st.dataframe(interventions[interventions.any(axis=1)])

series = series.dropna()
train, test = train_test_split_monthly(series, eval_start=eval_start, eval_end=eval_end)

st.write(f"Observaciones: Train={len(train)}, Test={len(test)}")

# Auto-sanar si es necesario
if robust_mode:
    if train.isna().any() or not np.isfinite(train.values).all():
        train = train.replace([np.inf, -np.inf], np.nan)
        train = train.interpolate(method="time").interpolate(method="linear").ffill().bfill()
        if train.isna().any() or not np.isfinite(train.values).all():
            med = float(np.nanmedian(train.values)) if np.isfinite(train.values).any() else 0.0
            train = train.fillna(med)
    if len(train) < int(min_train_months):
        st.warning(f"Train corto ({len(train)} meses). Se reduce la búsqueda (p,q<=1).")
        max_pq = min(max_pq, 1)
else:
    if len(train) < int(min_train_months):
        st.error(f"El conjunto de entrenamiento es demasiado corto (<{int(min_train_months)} meses). Ajusta la ventana o habilita Modo robusto.")
        st.stop()
    if train.isna().any() or not np.isfinite(train.values).all():
        st.error('El conjunto de entrenamiento contiene valores NaN/Inf. Habilita Modo robusto o ajusta la limpieza/ventana.')
        st.stop()

st.subheader("Búsqueda y selección de modelos")
with st.spinner("Entrenando ARIMA / SARIMA / SARIMAX..."):
    out = grid_search_models(train, test, seasonal_period=int(seasonal_period),
                             max_pq=int(max_pq), K_fourier=range(K_min, K_max+1),
                             mape_threshold=float(mape_thr), enforce_threshold=bool(enforce_thr))

if out["best"] is None or len(out["summary"]) == 0:
    st.error("No fue posible ajustar modelos válidos. Revisa los datos.")
    st.stop()

st.markdown("### Ranking de modelos")
display_cols = ["model", "order", "seasonal_order", "exog", "aic", "mape", "jb_p", "lb_p", "arch_p", "passes_all", "meets_thresh"]
summary_df = out["summary"].copy()
for c in display_cols:
    if c not in summary_df.columns:
        summary_df[c] = np.nan
st.dataframe(summary_df[display_cols].style.format({"aic":"{:.1f}", "mape":"{:.2f}%", "jb_p":"{:.3f}", "lb_p":"{:.3f}", "arch_p":"{:.3f}"}))

best = out["best"]
meets = (best.get('mape', 1e9) <= float(mape_thr))
st.success(f"**Mejor modelo:** {best['model']} | order={best['order']} | seasonal={best['seasonal_order']} | exog={best['exog']} | MAPE={best['mape']:.2f}% | Cumple MAPE ≤ {float(mape_thr):.2f}%: {'Sí' if meets else 'No'}")
if best.get("_note"):
    st.warning(best["_note"])

# Pronóstico (con respaldo)
def forecast_best(best, train, test, seasonal_period):
    if str(best["model"]).startswith("SARIMAX"):
        import re as _re
        m = _re.search(r"K=(\d+)", best["model"])
        k = int(m.group(1)) if m else 1
        exog_train = fourier_terms(train.index, period=int(seasonal_period), K=k)
        exog_test  = fourier_terms(test.index,  period=int(seasonal_period), K=k)
        res_tmp = SARIMAX(train, order=tuple(best["order"]), seasonal_order=tuple(best["seasonal_order"]),
                          exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = res_tmp.get_forecast(steps=len(test), exog=exog_test).predicted_mean
        fc.index = test.index
        return fc
    else:
        res_tmp = SARIMAX(train, order=tuple(best["order"]), seasonal_order=tuple(best["seasonal_order"]),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = res_tmp.get_forecast(steps=len(test)).predicted_mean
        fc.index = test.index
        return fc

fc = find_forecast_series(summary_df, best)
if fc is None:
    fc = forecast_best(best, train, test, int(seasonal_period))

st.markdown("### Pronóstico vs. Real (ventana de evaluación)")
plot_series(train, test, fc, "Pronóstico sobre la muestra de evaluación")

st.markdown("### Diagnósticos del mejor modelo")
st.write({
    "Jarque–Bera p-value (normalidad)": round(float(best["jb_p"]), 4),
    "Ljung–Box p-value (no autocorrelación)": round(float(best["lb_p"]), 4),
    "ARCH p-value (no heterocedasticidad)": round(float(best["arch_p"]), 4),
})

resid_best = refit_and_get_resid(best["model"], tuple(best["order"]), tuple(best["seasonal_order"]), train, int(seasonal_period))
plot_residuals(resid_best, title_prefix=f"{best['model']}")

with st.expander("SUMMARY del mejor modelo (statsmodels)", expanded=False):
    try:
        res_best = fit_best_model_for_summary(best, train, int(seasonal_period))
        st.text(res_best.summary().as_text())
    except Exception as e:
        st.warning(f"No se pudo generar el SUMMARY del mejor modelo: {e}")

st.markdown("### Exportar resultados")
meets_json = (best.get('mape', 1e9) <= float(mape_thr))
export = {
    "best_model": {
        "model": best["model"],
        "order": tuple(best["order"]),
        "seasonal_order": tuple(best["seasonal_order"]),
        "exog": best["exog"],
        "aic": float(best["aic"]),
        "mape_eval_%": float(best["mape"]),
        "meets_mape_threshold": bool(meets_json),
        "note": best.get("_note"),
        "diagnostics_pvalues": {
            "jarque_bera": float(best["jb_p"]),
            "ljung_box": float(best["lb_p"]),
            "arch": float(best["arch_p"])
        }
    },
    "evaluation_window": {"test_start": str(test.index.min()), "test_end": str(test.index.max())},
}

json_bytes = io.BytesIO()
json_bytes.write(pd.Series(export).to_json().encode("utf-8"))
json_bytes.seek(0)
st.download_button("Descargar resumen JSON", data=json_bytes, file_name="soya_model_summary.json", mime="application/json")

csv_bytes = io.BytesIO()
pd.DataFrame({"real": test, "forecast": fc}).to_csv(csv_bytes, index=True)
csv_bytes.seek(0)
st.download_button("Descargar pronóstico (CSV)", data=csv_bytes, file_name="forecast_eval.csv", mime="text/csv")
