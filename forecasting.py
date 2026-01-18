import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Optional forecasters – used when available; skipped otherwise.
try:  # pragma: no cover - optional dependency
    from prophet import Prophet  # type: ignore

    HAS_PROPHET = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PROPHET = False

try:  # pragma: no cover - optional dependency
    import xgboost as xgb  # type: ignore

    HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGB = False


def _logistic_fn(x, L, k, x0):
    """
    Standard logistic function used for curve fitting.
    """
    return L / (1 + np.exp(-k * (x - x0)))


@dataclass
class ForecastResult:
    """
    Lightweight container for an individual model forecast.
    """
    name: str
    value: float


class ForecastEnsembler:
    """
    Combines multiple simple classical forecasting methods.
    Returns the average of all successful models.
    """

    def __init__(self):
        self.last_models: List[ForecastResult] = []

    def _prep_history(self, df: pd.DataFrame, cur_time: float, history_horizon: int = 6 * 3600) -> pd.DataFrame:
        """
        Select a rolling history window to fit models on.
        """
        history = df[df["Time"] <= cur_time]
        if history.empty:
            history = df.copy()
        lower_bound = max(history["Time"].max() - history_horizon, history["Time"].min())
        history = history[history["Time"] >= lower_bound]
        return history

    def _fit_linear_regression(self, x: np.ndarray, y: np.ndarray, target_time: float) -> float:
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        return float(model.predict(np.array([[target_time]]))[0])

    def _fit_gradient_boosting(self, x: np.ndarray, y: np.ndarray, target_time: float) -> float:
        if len(x) < 3:
            return float(np.mean(y))
        if HAS_XGB:  # pragma: no cover - optional dependency
            booster = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
            )
            booster.fit(x.reshape(-1, 1), y)
            pred = booster.predict(np.array([[target_time]]))[0]
        else:
            booster = GradientBoostingRegressor(random_state=0)
            booster.fit(x.reshape(-1, 1), y)
            pred = booster.predict(np.array([[target_time]]))[0]
        return float(pred)

    def _fit_exponential_smoothing(self, y: np.ndarray) -> float:
        if len(y) < 2:
            return float(y[-1])
        model = SimpleExpSmoothing(y, initialization_method="heuristic").fit()
        return float(model.forecast(1)[0])

    def _fit_logistic_curve(self, x: np.ndarray, y: np.ndarray, target_time: float) -> float:
        if len(x) < 4:
            return float(np.mean(y))
        # Normalize times to improve stability.
        x_norm = x - x.min()
        try:
            params, _ = curve_fit(
                _logistic_fn,
                x_norm,
                y,
                p0=[max(y), 1, np.median(x_norm)],
                maxfev=10000,
            )
            L, k, x0 = params
            return float(_logistic_fn(target_time - x.min(), L, k, x0))
        except Exception:
            return float(np.mean(y))

    def _fit_prophet(self, df: pd.DataFrame, target_time: float) -> float:
        if not HAS_PROPHET:  # pragma: no cover - optional dependency
            raise ImportError("Prophet not installed")
        if len(df) < 10:
            raise ValueError("Insufficient data for Prophet")
        prophet_df = pd.DataFrame({"ds": pd.to_datetime(df["Time"], unit="s"), "y": df["Predicted"]})
        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
        model.fit(prophet_df)
        future = pd.DataFrame({"ds": [pd.to_datetime(target_time, unit="s")]})
        forecast = model.predict(future)
        return float(forecast["yhat"].iloc[0])

    def forecast_window(
        self,
        df: pd.DataFrame,
        cur_time: float,
        window_start: float,
        window_end: float,
    ) -> Tuple[float, List[ForecastResult]]:
        """
        Produce an ensemble forecast for a future window.
        Returns the averaged forecast and the individual model outputs.
        """
        self.last_models = []
        if df.empty:
            return 0.0, self.last_models

        history = self._prep_history(df, cur_time)
        x_hist = history["Time"].to_numpy()
        y_hist = history["Predicted"].to_numpy()
        target_time = window_end

        # 1) ARIMA output already provided in the forecast CSV (take max in window)
        future_window = df[(df["Time"] >= window_start) & (df["Time"] <= window_end)]
        if not future_window.empty:
            arima_pred = float(future_window["Predicted"].max())
            self.last_models.append(ForecastResult("arima_file", arima_pred))

        # 2) Linear regression
        try:
            lin_pred = self._fit_linear_regression(x_hist, y_hist, target_time)
            self.last_models.append(ForecastResult("linear_regression", lin_pred))
        except Exception as exc:
            logging.debug(f"Linear regression forecast failed: {exc}")

        # 3) Exponential smoothing
        try:
            exp_pred = self._fit_exponential_smoothing(y_hist)
            self.last_models.append(ForecastResult("exp_smoothing", exp_pred))
        except Exception as exc:
            logging.debug(f"Exponential smoothing forecast failed: {exc}")

        # 4) Moving average (simple baseline)
        ma_pred = float(np.mean(y_hist[-10:])) if len(y_hist) > 0 else 0.0
        self.last_models.append(ForecastResult("moving_average", ma_pred))

        # 5) Gradient boosting / XGBoost-style regressor
        try:
            gb_pred = self._fit_gradient_boosting(x_hist, y_hist, target_time)
            self.last_models.append(ForecastResult("boosting_regressor", gb_pred))
        except Exception as exc:
            logging.debug(f"Boosting forecast failed: {exc}")

        # 6) Logistic curve regression
        try:
            log_pred = self._fit_logistic_curve(x_hist, y_hist, target_time)
            self.last_models.append(ForecastResult("logistic_curve", log_pred))
        except Exception as exc:
            logging.debug(f"Logistic forecast failed: {exc}")

        # 7) Prophet (optional)
        if HAS_PROPHET:
            try:
                prophet_pred = self._fit_prophet(history, target_time)
                self.last_models.append(ForecastResult("prophet", prophet_pred))
            except Exception as exc:
                logging.debug(f"Prophet forecast skipped: {exc}")

        values = [m.value for m in self.last_models if np.isfinite(m.value)]
        if len(values) == 0:
            return 0.0, self.last_models
        return float(np.mean(values)), self.last_models
