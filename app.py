
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

# ----------------------------
# Cleaning helpers
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
        # require positive values
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


st.set_page_config(page_title="Soya ARIMA/SARIMA/SARIMAX Selector", layout="wide")

# ----------------------------
# Helpers
# ----------------------------

def try_parse_dates(s: pd.Series) -> pd.Series:
    # Try multiple common formats, force dayfirst
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def find_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in ["fecha", "date", "period", "time", "month", "fecha_mes"]]
    # If none of the common names, try to parse columns by dtype
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
    # If frequency is daily or irregular, aggregate to month-end mean
    s = series.dropna().sort_index()
    if s.index.inferred_freq in [None, "D", "B", "W", "QS", "Q", "A", "H", "T", "S"]:
        return s.resample("M").mean().dropna()
    # If it's monthly but not aligned to month-end, align
    if s.index.inferred_freq and "M" in s.index.inferred_freq:
        return s.asfreq("M")
    # Fallback to month-end
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
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
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
    # Generate Fourier seasonal terms for SARIMAX exog
    t = np.arange(len(index))
    X = {}
    for k in range(1, K+1):
        X[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        X[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(X, index=index)

def select_differencing(y: pd.Series) -> int:
    # Robust ADF: handle short/constant/invalid series gracefully
    y_clean = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(y_clean) < 12:
        # too short for a reliable unit root test; avoid differencing by default
        return 0
    if y_clean.std() == 0 or y_clean.max() == y_clean.min():
        # constant series -> no differencing
        return 0
    try:
        adf_p = adfuller(y_clean.values, autolag="AIC")[1]
        return 1 if adf_p > 0.05 else 0
    except Exception:
        # fall back safely
        return 0

def diagnostics(res, y, exog=None) -> Dict[str, float]:
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
        "mape": mape(test.values, fc.values),
        "jb_p": diag["jb_p"],
        "lb_p": diag["lb_p"],
        "arch_p": diag["arch_p"],
        "forecast": fc
    }

def grid_search_models(train: pd.Series, test: pd.Series, seasonal_period=12, max_pq=3, K_fourier=(1,2,3), mape_threshold=None, enforce_threshold=False) -> Dict[str, Any]:
    results = []

    d = select_differencing(train)
    # ARIMA (no seasonal)
    for p in range(0, max_pq+1):
        for q in range(0, max_pq+1):
            try:
                res = fit_sarimax(train, order=(p,d,q))
                fc = res.get_forecast(steps=len(test)).predicted_mean
                fc.index = test.index
                diag = diagnostics(res, train)
                results.append(record_result("ARIMA", (p,d,q), (0,0,0,0), None, test, fc, diag, res.aic))
            except Exception as e:
                continue

    # SARIMA (seasonal part)
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

    # SARIMAX (with Fourier terms)
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
                        diag = diagnostics(res, train, exog_train)
                        results.append(record_result(f"SARIMAX(K={K})", (p,d,q), (p,D,q,seasonal_period), f"Fourier K={K}", test, fc, diag, res.aic))
                    except Exception:
                        continue

    # Rank: first, filter those that pass all diagnostics; then by MAPE; if none pass, sort by number of passes then MAPE
    df = pd.DataFrame(results)
    if len(df)==0:
        return {"summary": pd.DataFrame(), "best": None}

    df["passes_all"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).all(axis=1)
    df["passes_count"] = df[["jb_p", "lb_p", "arch_p"]].ge(0.05).sum(axis=1)

    candidates = df[df["passes_all"]].copy()
    if len(candidates) > 0:
        best_row = candidates.sort_values(["mape", "aic"]).iloc[0].to_dict()
    else:
        best_row = df.sort_values(["passes_count", "mape", "aic"], ascending=[False, True, True]).iloc[0].to_dict()
    return {"summary": df.sort_values(["passes_all","passes_count","mape","aic"], ascending=[False, False, True, True]), "best": best_row}

def plot_series(train, test, fc, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test (holdout)")
    fc.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def plot_residuals(resid, title_prefix=""):

def find_forecast_series(summary_df, best_dict):
    mask = (summary_df["model"]==best_dict["model"]) & \
           (summary_df["order"]==tuple(best_dict["order"])) & \
           (summary_df["seasonal_order"]==tuple(best_dict["seasonal_order"])) & \
           (summary_df["mape"]==best_dict["mape"])
    subset = summary_df[mask]
    if len(subset)==0:
        # fallback: take the first row
        subset = summary_df.iloc[[0]]
    row = subset.iloc[0]
    return row.get("forecast")

fc = find_forecast_series(out["summary"], best)

st.markdown("### Pronóstico vs. Real (ventana de evaluación)")
plot_series(train, test, fc, "Pronóstico sobre la muestra de evaluación")

# Residuals diagnostics for the best model
st.markdown("### Diagnósticos del mejor modelo")
st.write({
    "Jarque–Bera p-value (normalidad)": round(float(best["jb_p"]), 4),
    "Ljung–Box p-value (no autocorrelación)": round(float(best["lb_p"]), 4),
    "ARCH p-value (no heterocedasticidad)": round(float(best["arch_p"]), 4),
})

# Refit the best model to combine train+test for residuals visualization (optional)
# For residuals during train fit:
def refit_and_get_resid(model_name, order, seasonal_order, train, seasonal_period, exog_desc):
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

resid_best = refit_and_get_resid(best["model"], tuple(best["order"]), tuple(best["seasonal_order"]), train, int(seasonal_period), best["exog"])
plot_residuals(resid_best, title_prefix=f"{best['model']}")


def fit_best_model_for_summary(best: dict, train: pd.Series, seasonal_period: int):
    """Refit del mejor modelo sobre el conjunto de entrenamiento para obtener .summary()."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import re as _re

    model_name = str(best.get("model", ""))
    order = tuple(best.get("order", (0, 0, 0)))
    seasonal_order = tuple(best.get("seasonal_order", (0, 0, 0, 0)))

    if model_name.startswith("SARIMAX"):
        m = _re.search(r"K=(\d+)", model_name)
        k = int(m.group(1)) if m else 1
        exog_train = fourier_terms(train.index, period=int(seasonal_period), K=k)
        res = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    else:
        res = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    return res

with st.expander("SUMMARY del mejor modelo (statsmodels)", expanded=False):
    try:
        res_best = fit_best_model_for_summary(best, train, int(seasonal_period))
        if res_best is not None:
            st.text(res_best.summary().as_text())
        else:
            st.warning("No se pudo generar el SUMMARY del mejor modelo (res_best=None).")
    except Exception as e:
        st.warning(f"No se pudo generar el SUMMARY del mejor modelo: {e}")


# Export results
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
