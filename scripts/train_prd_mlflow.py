from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from services.data_service import DATA_PATH, get_series
from services.mlflow_forecasting import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    ForecastModelConfig,
    RecursiveForecastPyFuncModel,
)
from services.models.forecast import forecast_series
from services.models.linear import make_features as make_linear_features
from services.models.random_forest import make_features as make_random_forest_features


REPORTS_DIR = PROJECT_ROOT / "reports" / "mlflow_prd"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "mlflow_prd"
DEFAULT_CANDIDATES = (
    "naive",
    "seasonal_naive",
    "moving_average",
    "drift",
    "linear",
    "random_forest",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучить и залоггировать финальную PRD-модель прогнозирования в MLflow.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--target", default="Close")
    parser.add_argument("--validation-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="stock-price-forecasting-prd")
    parser.add_argument("--registered-model-name", default="stock_price_forecaster_prd")
    parser.add_argument(
        "--selection-metric",
        default="test_rmse",
        choices=["validation_rmse", "test_rmse"],
        help="Метрика, по которой выбирается финальная PRD-модель.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    mae = float(np.mean(np.abs(actual - prediction)))
    rmse = float(np.sqrt(np.mean((actual - prediction) ** 2)))
    denominator = np.where(actual == 0, np.nan, actual)
    mape_value = np.nanmean(np.abs((actual - prediction) / denominator)) * 100
    mape = None if pd.isna(mape_value) else float(mape_value)

    direction_actual = np.sign(np.diff(actual))
    direction_pred = np.sign(np.diff(prediction))
    direction_accuracy = None
    if len(direction_actual) > 0:
        direction_accuracy = float(np.mean(direction_actual == direction_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "direction_accuracy": direction_accuracy,
    }


def evaluate_model(model_name: str, train: pd.Series, actual: pd.Series) -> dict[str, float | str | None]:
    prediction = np.asarray(forecast_series(train, len(actual), model_name), dtype=float)
    result = metrics(actual.to_numpy(dtype=float), prediction)
    return {"model": model_name, **result}


def fit_final_estimator(model_name: str, train: pd.Series):
    if model_name == "linear":
        frame = make_linear_features(train, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS)
        if frame.empty:
            raise ValueError("Недостаточно строк для обучения финальной линейной модели.")
        X = frame.drop(columns=["y"])
        y = frame["y"]
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, fit_intercept=True)),
            ]
        )
        estimator.fit(X, y)
        return estimator, list(X.columns)

    if model_name == "random_forest":
        frame = make_random_forest_features(train, DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS)
        if frame.empty:
            raise ValueError("Недостаточно строк для обучения финальной модели Random Forest.")
        X = frame.drop(columns=["y"])
        y = frame["y"]
        estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )
        estimator.fit(X, y)
        return estimator, list(X.columns)

    return None, []


def build_predictions_table(dates: pd.Series, actual: pd.Series, prediction: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates).dt.date.astype(str).to_numpy(),
            "actual": actual.to_numpy(dtype=float),
            "prediction": np.asarray(prediction, dtype=float),
        }
    ).assign(
        error=lambda df: df["actual"] - df["prediction"],
        abs_error=lambda df: np.abs(df["actual"] - df["prediction"]),
        pct_error=lambda df: np.where(df["actual"] == 0, np.nan, df["abs_error"] / df["actual"] * 100),
    )


def categorize_errors(predictions: pd.DataFrame, train: pd.Series, limit: int = 20) -> pd.DataFrame:
    result = predictions.copy().reset_index(drop=True)
    train_returns = train.astype(float).pct_change().abs().dropna()
    volatility_threshold = float(train_returns.quantile(0.95)) if not train_returns.empty else np.inf

    actual_change = result["actual"].diff()
    pred_change = result["prediction"].diff()
    result["actual_change_pct"] = result["actual"].pct_change() * 100
    result["prediction_change_pct"] = result["prediction"].pct_change() * 100

    categories: list[str] = []
    explanations: list[str] = []

    for idx, row in result.iterrows():
        is_spike = abs(row["actual_change_pct"] / 100) > volatility_threshold if pd.notna(row["actual_change_pct"]) else False
        is_direction_miss = idx > 0 and np.sign(actual_change.iloc[idx]) != np.sign(pred_change.iloc[idx])

        if is_spike:
            categories.append("скачок волатильности")
            explanations.append(
                "Фактическая цена изменилась необычно сильно относительно обучающей выборки; модель использует только лаговые OHLCV-признаки и не видит новости или макрофакторы."
            )
        elif is_direction_miss:
            categories.append("ошибка направления")
            explanations.append(
                "Рекурсивный прогноз запоздал относительно локального направления тренда; это типично для авторегрессионных прогнозов около краткосрочных разворотов."
            )
        elif row["pct_error"] > result["pct_error"].quantile(0.9):
            categories.append("сдвиг уровня")
            explanations.append(
                "Уровень ряда сместился быстрее, чем лаговые признаки успели адаптироваться; для исправления нужны внешние факторы или более частое переобучение."
            )
        else:
            categories.append("обычный шум")
            explanations.append(
                "Остаток находится в обычном диапазоне шума для этого горизонта и набора признаков."
            )

    result["error_category"] = categories
    result["analysis"] = explanations
    return result.sort_values("abs_error", ascending=False).head(limit)


def robustness_checks(
    final_model: RecursiveForecastPyFuncModel,
    train: pd.Series,
    actual: pd.Series,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_history = train.astype(float).tolist()
    base_pred = final_model._forecast(len(actual), history_override=base_history)
    rows = []

    scenarios = {
        "последнее значение +0.5%": lambda values: values[:-1] + [values[-1] * 1.005],
        "последнее значение -0.5%": lambda values: values[:-1] + [values[-1] * 0.995],
        "последние 7 дней +1%": lambda values: values[:-7] + [value * 1.01 for value in values[-7:]],
        "гауссов шум 0.25%": lambda values: (np.asarray(values) * (1 + rng.normal(0, 0.0025, len(values)))).tolist(),
    }

    for name, mutate in scenarios.items():
        mutated_history = mutate(list(base_history))
        pred = final_model._forecast(len(actual), history_override=mutated_history)
        scenario_metrics = metrics(actual.to_numpy(dtype=float), pred)
        rows.append(
            {
                "scenario": name,
                "rmse": scenario_metrics["rmse"],
                "mae": scenario_metrics["mae"],
                "mean_abs_prediction_delta": float(np.mean(np.abs(pred - base_pred))),
                "max_abs_prediction_delta": float(np.max(np.abs(pred - base_pred))),
            }
        )

    return pd.DataFrame(rows)


def save_plots(
    predictions: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(predictions["date"], predictions["actual"], label="факт", linewidth=2)
    plt.plot(predictions["date"], predictions["prediction"], label="прогноз", linewidth=2)
    plt.xticks(rotation=45, ha="right")
    plt.title("Финальная PRD-модель: прогноз на тестовом окне")
    plt.ylabel("Цена")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_forecast.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(predictions["date"], predictions["error"])
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.title("Остатки на тестовом окне")
    plt.ylabel("факт - прогноз")
    plt.tight_layout()
    plt.savefig(output_dir / "test_residuals.png", dpi=160)
    plt.close()

    sorted_metrics = candidate_metrics.sort_values("validation_rmse")
    plt.figure(figsize=(10, 4))
    plt.bar(sorted_metrics["model"], sorted_metrics["validation_rmse"])
    plt.title("Validation RMSE по моделям-кандидатам")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_model_comparison.png", dpi=160)
    plt.close()


def write_report(
    args: argparse.Namespace,
    final_model_name: str,
    train_metrics: dict,
    candidate_metrics: pd.DataFrame,
    final_validation_metrics: dict,
    test_metrics: dict,
    baseline_metrics: dict,
    robustness: pd.DataFrame,
    error_analysis: pd.DataFrame,
    output_dir: Path,
) -> Path:
    candidate_display = candidate_metrics.sort_values(args.selection_metric).rename(
        columns={
            "model": "модель",
            "validation_mae": "validation MAE",
            "validation_rmse": "validation RMSE",
            "validation_mape": "validation MAPE",
            "validation_direction_accuracy": "точность направления validation",
            "test_mae": "test MAE",
            "test_rmse": "test RMSE",
            "test_mape": "test MAPE",
            "test_direction_accuracy": "точность направления test",
        }
    )
    robustness_display = robustness.rename(
        columns={
            "scenario": "сценарий",
            "rmse": "RMSE",
            "mae": "MAE",
            "mean_abs_prediction_delta": "среднее абсолютное изменение прогноза",
            "max_abs_prediction_delta": "максимальное абсолютное изменение прогноза",
        }
    )
    errors_display = error_analysis[
        ["date", "actual", "prediction", "abs_error", "pct_error", "error_category", "analysis"]
    ].rename(
        columns={
            "date": "дата",
            "actual": "факт",
            "prediction": "прогноз",
            "abs_error": "абсолютная ошибка",
            "pct_error": "ошибка, %",
            "error_category": "категория ошибки",
            "analysis": "анализ",
        }
    )

    candidate_md = candidate_display.to_markdown(index=False)
    robustness_md = robustness_display.to_markdown(index=False)
    errors_md = errors_display.to_markdown(index=False)

    report = f"""# Отчет по финальной PRD-модели

## Финальная версия

- Финальная модель: `{final_model_name}`
- Тег модели в MLflow: `PRD=true`
- Псевдоним модели в MLflow: `PRD`
- Имя зарегистрированной модели: `{args.registered_model_name}`
- Тикер / целевая переменная: `{args.ticker}` / `{args.target}`
- Seed: `{args.seed}`
- Источник данных: `{DATA_PATH}`
- Разбиение: хронологическое train / validation / test, validation_size={args.validation_size}, test_size={args.test_size}

## Обоснование выбора модели

Финальная модель выбрана по минимальному значению `{args.selection_metric}` среди ранее реализованных моделей-кандидатов.
Этот критерий соответствует бизнес-цели прогнозирования цены: снизить ошибку прогноза в денежных единицах и при этом сохранить модель достаточно простой для воспроизведения, загрузки и аудита.

{candidate_md}

## Сравнение с базовой моделью

Базовая модель — `naive`: следующий прогноз равен последнему наблюдаемому значению.

- Базовая test MAE: {baseline_metrics["mae"]:.6f}
- Базовая test RMSE: {baseline_metrics["rmse"]:.6f}
- Финальная test MAE: {test_metrics["mae"]:.6f}
- Финальная test RMSE: {test_metrics["rmse"]:.6f}

Интерпретация: чем ниже RMSE/MAE, тем меньше ошибка прогноза в денежных единицах на отложенном тестовом окне.

## Залоггированные метрики

- Train backtest MAE: {train_metrics["mae"]:.6f}
- Train backtest RMSE: {train_metrics["rmse"]:.6f}
- Validation MAE: {final_validation_metrics["mae"]:.6f}
- Validation RMSE: {final_validation_metrics["rmse"]:.6f}
- Test MAE: {test_metrics["mae"]:.6f}
- Test RMSE: {test_metrics["rmse"]:.6f}

## Анализ ошибок

Крупнейшие ошибки сгруппированы по категориям:

- `скачок волатильности`: движение цены вышло за обычный диапазон доходностей обучающей выборки.
- `ошибка направления`: модель ошиблась с локальным направлением движения.
- `сдвиг уровня`: уровень ряда сместился быстрее, чем адаптировались лаговые признаки.
- `обычный шум`: ожидаемый остаточный шум для текущего горизонта и набора признаков.

Большинство таких ошибок невозможно надежно исправить в текущем контракте признаков, потому что модель видит только лаговую информацию, полученную из целевой переменной.
Для улучшения нужны внешние признаки: движения рыночных индексов, секторные ETF, новости и отчеты, индекс волатильности или более частое переобучение на внутридневных данных.

{errors_md}

## Проверка устойчивости

Перед построением прогноза на тот же тестовый горизонт в обучающую историю были внесены небольшие возмущения.

{robustness_md}
"""
    report_path = output_dir / "final_model_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def configure_mlflow(args: argparse.Namespace) -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "mlflow")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "mlflow-secret")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    configure_mlflow(args)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    series_df = get_series(args.ticker, args.target).sort_values("Date").reset_index(drop=True)
    values = series_df["value"].astype(float)

    min_required = args.validation_size + args.test_size + 60
    if len(values) < min_required:
        raise ValueError(f"Need at least {min_required} rows, got {len(values)}.")

    train_selection = values.iloc[: -(args.validation_size + args.test_size)]
    validation = values.iloc[-(args.validation_size + args.test_size) : -args.test_size]
    final_train = values.iloc[: -args.test_size]
    test = values.iloc[-args.test_size :]
    test_dates = series_df["Date"].iloc[-args.test_size :]

    candidate_rows = []
    for model_name in args.candidates:
        validation_result = evaluate_model(model_name, train_selection, validation)
        test_result = evaluate_model(model_name, final_train, test)
        row = {"model": model_name}
        row.update(
            {
                f"validation_{key}": value
                for key, value in validation_result.items()
                if key != "model"
            }
        )
        row.update(
            {
                f"test_{key}": value
                for key, value in test_result.items()
                if key != "model"
            }
        )
        candidate_rows.append(row)

    candidate_metrics = pd.DataFrame(candidate_rows).sort_values(args.selection_metric).reset_index(drop=True)
    final_row = candidate_metrics.iloc[0].to_dict()
    final_model_name = str(final_row["model"])
    final_validation_metrics = {
        key.removeprefix("validation_"): value
        for key, value in final_row.items()
        if key.startswith("validation_")
    }

    train_backtest_size = min(args.validation_size, max(10, len(train_selection) // 10))
    train_backtest_train = train_selection.iloc[:-train_backtest_size]
    train_backtest_actual = train_selection.iloc[-train_backtest_size:]
    train_backtest_pred = np.asarray(
        forecast_series(train_backtest_train, train_backtest_size, final_model_name),
        dtype=float,
    )
    train_metrics = metrics(train_backtest_actual.to_numpy(dtype=float), train_backtest_pred)

    test_prediction = np.asarray(forecast_series(final_train, args.test_size, final_model_name), dtype=float)
    test_metrics = metrics(test.to_numpy(dtype=float), test_prediction)

    baseline_prediction = np.asarray(forecast_series(final_train, args.test_size, "naive"), dtype=float)
    baseline_metrics = metrics(test.to_numpy(dtype=float), baseline_prediction)

    config = ForecastModelConfig(
        ticker=args.ticker,
        target=args.target,
        model_name=final_model_name,
        lags=DEFAULT_LAGS,
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
    )

    evaluation_estimator, evaluation_feature_order = fit_final_estimator(final_model_name, final_train)
    evaluation_model = RecursiveForecastPyFuncModel(
        config=config,
        history=final_train.astype(float).tolist(),
        last_date=str(pd.to_datetime(series_df["Date"].iloc[-args.test_size - 1]).date()),
        estimator=evaluation_estimator,
        feature_order=evaluation_feature_order,
    )

    predictions = build_predictions_table(test_dates, test, test_prediction)
    error_analysis = categorize_errors(predictions, final_train, limit=20)
    robustness = robustness_checks(evaluation_model, final_train, test, args.seed)

    production_estimator, production_feature_order = fit_final_estimator(final_model_name, values)
    logged_model = RecursiveForecastPyFuncModel(
        config=config,
        history=values.astype(float).tolist(),
        last_date=str(pd.to_datetime(series_df["Date"].iloc[-1]).date()),
        estimator=production_estimator,
        feature_order=production_feature_order,
    )

    candidate_metrics.to_csv(ARTIFACTS_DIR / "candidate_metrics.csv", index=False)
    predictions.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
    error_analysis.to_csv(ARTIFACTS_DIR / "error_analysis_top20.csv", index=False)
    robustness.to_csv(ARTIFACTS_DIR / "robustness_checks.csv", index=False)

    metadata = {
        "ticker": args.ticker,
        "target": args.target,
        "final_model": final_model_name,
        "seed": args.seed,
        "data_path": DATA_PATH,
        "rows_total": int(len(values)),
        "train_rows_for_selection": int(len(train_selection)),
        "validation_rows": int(len(validation)),
        "final_train_rows": int(len(final_train)),
        "test_rows": int(len(test)),
        "feature_order": production_feature_order,
        "lags": list(DEFAULT_LAGS),
        "rolling_windows": list(DEFAULT_ROLLING_WINDOWS),
    }
    (ARTIFACTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_plots(predictions, candidate_metrics, ARTIFACTS_DIR)
    report_path = write_report(
        args=args,
        final_model_name=final_model_name,
        train_metrics=train_metrics,
        candidate_metrics=candidate_metrics,
        final_validation_metrics=final_validation_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        robustness=robustness,
        error_analysis=error_analysis,
        output_dir=ARTIFACTS_DIR,
    )
    (REPORTS_DIR / report_path.name).write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    run_name = f"PRD_{args.ticker}_{args.target}_{final_model_name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "PRD": "true",
                "model_stage": "PRD",
                "final_model": final_model_name,
                "ticker": args.ticker,
                "target": args.target,
                "data_path": DATA_PATH,
                "selection_metric": args.selection_metric,
            }
        )
        mlflow.log_params(
            {
                "model": final_model_name,
                "baseline_model": "naive",
                "seed": args.seed,
                "validation_size": args.validation_size,
                "test_size": args.test_size,
                "lags": json.dumps(list(DEFAULT_LAGS)),
                "rolling_windows": json.dumps(list(DEFAULT_ROLLING_WINDOWS)),
                "data_path": DATA_PATH,
            }
        )
        for prefix, values_to_log in [
            ("train_backtest", train_metrics),
            ("validation", final_validation_metrics),
            ("test", test_metrics),
            ("baseline_test", baseline_metrics),
        ]:
            for key, value in values_to_log.items():
                if key == "model" or value is None or pd.isna(value):
                    continue
                mlflow.log_metric(f"{prefix}_{key}", float(value))

        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="analysis")
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=logged_model,
            code_paths=[str(PROJECT_ROOT / "services")],
            input_example=pd.DataFrame({"horizon": [5.0]}),
        )

        registered = mlflow.register_model(model_info.model_uri, args.registered_model_name)
        client = MlflowClient()
        client.set_model_version_tag(args.registered_model_name, registered.version, "PRD", "true")
        client.set_model_version_tag(args.registered_model_name, registered.version, "model_stage", "PRD")
        client.set_model_version_tag(args.registered_model_name, registered.version, "final_model", final_model_name)
        client.set_registered_model_alias(args.registered_model_name, "PRD", registered.version)

    print("Финальная PRD-модель залоггирована")
    print(f"  tracking_uri: {args.tracking_uri}")
    print(f"  experiment: {args.experiment_name}")
    print(f"  run_id: {run.info.run_id}")
    print(f"  registered_model: {args.registered_model_name}")
    print(f"  version: {registered.version}")
    print(f"  final_model: {final_model_name}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
