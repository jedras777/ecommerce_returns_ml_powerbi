from pathlib import Path
import pandas as pd
import yaml

from EDA.data_preparer import DataPreparer
from EDA.eda_analysis import DataQuality
from EDA.business_eda import BusinessEDA
from ML.aggregation import FeatureEngineer
from ML.multi_model_trainer import MultiModelTrainer
from ML.visualizations import Visualizer


def load_config() -> dict:
    root = Path(__file__).resolve().parent
    cfg_path = root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # make paths absolute
    for k, v in cfg.get("paths", {}).items():
        cfg["paths"][k] = str((root / v).resolve())
    return cfg


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config()
    p = cfg["paths"]
    t = cfg["training"]

    # === 0) Data prep: Excel -> raw CSV -> clean CSV (idempotent) ===
    prep = DataPreparer(
        excel_path=p["inputs"],
        csv_path=p["raw_csv"],
        clean_path=p["clean_csv"],
    )
    if not Path(p["raw_csv"]).exists():
        ensure_parent(p["raw_csv"])
        prep.excel_to_csv()
    if not Path(p["clean_csv"]).exists():
        ensure_parent(p["clean_csv"])
        prep.clean_data()

    # === 1) Quality EDA (raw vs clean) ===
    dq = DataQuality(raw_csv_path=Path(p["raw_csv"]), clean_csv_path=Path(p["clean_csv"]))
    dq.run()  # prints summary + comparison

    # === 2) Business EDA ===
    df = pd.read_csv(p["clean_csv"], parse_dates=["InvoiceDate"])
    beda = BusinessEDA(df)
    beda.sales_over_time()
    beda.returns_analysis()
    beda.products_analysis()
    beda.rfm_analysis()

    # === 3) Feature Engineering + chronological split ===
    fe = FeatureEngineer(clean_path=p["clean_csv"])
    basket, X, y, X_train, X_test, y_train, y_test = fe.prepare(cutoff=t["cutoff_date"])

    # === 4) Train multiple models & evaluate ===
    mt = MultiModelTrainer(X_train, X_test, y_train, y_test, basket=basket)
    mt.build_models(names=tuple(t.get("models", ["logreg", "rf", "xgb"])))
    mt.fit_all()
    metrics = mt.evaluate()
    mt.print_reports()

    # === 5) Export metrics & predictions (best model by F1) ===
    ensure_parent(p["metrics"])
    ensure_parent(p["predictions"])
    mt.export_metrics(p["metrics"])
    mt.export_predictions(p["predictions"])

    # === 6) Plots: multi-model ROC/PR, confusions, XGB importance ===
    payload = mt.get_eval_payload()
    viz = Visualizer(
        y_test=payload["y_test"],
        probas=payload["probas"],
        preds=payload["preds"],
        models=payload["models"],
    )
    out_dir = Path(p["metrics"]).parent  # reuse outputs folder
    viz.plot_roc(save_path=str(out_dir / "roc_models.png"))
    viz.plot_pr(save_path=str(out_dir / "pr_models.png"))
    viz.plot_confusions(save_path=str(out_dir / "confusions.png"))
    viz.plot_xgb_importance(save_path=str(out_dir / "feature_importance_xgb.png"))

    print("\nPipeline finished ✅")
    print(metrics)


if __name__ == "__main__":
    main()
