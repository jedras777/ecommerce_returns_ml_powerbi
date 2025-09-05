import pandas as pd
from pathlib import Path
from typing import Any

RAW_CSV_PATH: Path = Path(r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\outputs\online_retail.csv")
CLEAN_CSV_PATH: Path = Path(r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\outputs\online_retail_clean.csv")


def check_parameters(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Pretty console report with head / info / describe / missing + a few quick stats."""
    print(f"\n\n===== {name} =====")
    print("\nHead (5 rows):")
    print(df.head(5))

    print("\nInfo:")
    # df.info() prints to stdout; avoid wrapping in print() to prevent trailing 'None'
    df.info()

    print("\nDescribe (numeric columns):")
    print(df.describe())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Extra quick stats if columns exist
    cols = df.columns
    if "CustomerID" in cols:
        print("\nUnique customers:", df["CustomerID"].nunique(dropna=True))
    if "StockCode" in cols:
        print("Unique products:", df["StockCode"].nunique(dropna=True))
    if "InvoiceNo" in cols:
        print("Unique invoices:", df["InvoiceNo"].nunique(dropna=True))
    if "InvoiceDate" in cols and pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        print("Date range:", df["InvoiceDate"].min(), "→", df["InvoiceDate"].max())


def _safe_string(series: pd.Series) -> pd.Series:
    """Ensure series is string dtype for string ops (like .str.startswith)."""
    if not pd.api.types.is_string_dtype(series):
        return series.astype("string")
    return series


def data_quality_metrics(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Compute key data-quality and business sanity metrics for Online Retail.
    Returns a pandas Series with named metrics.
    """
    s = pd.Series(dtype="object")  # allow mixed types (ints/floats/datetimes)

    s["rows"] = len(df)
    s["cols"] = df.shape[1]

    # Ensure types for checks (non-destructive)
    if "InvoiceDate" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df = df.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    invoice_no = _safe_string(df["InvoiceNo"]) if "InvoiceNo" in df.columns else None
    description = _safe_string(df["Description"]) if "Description" in df.columns else None
    customer_id = df["CustomerID"] if "CustomerID" in df.columns else None

    # Date range
    if "InvoiceDate" in df.columns:
        s["date_min"] = pd.to_datetime(df["InvoiceDate"]).min()
        s["date_max"] = pd.to_datetime(df["InvoiceDate"]).max()

    # Core counts
    s["unique_customers"] = df["CustomerID"].nunique(dropna=True) if "CustomerID" in df.columns else pd.NA
    s["unique_products"]  = df["StockCode"].nunique(dropna=True)  if "StockCode"  in df.columns else pd.NA
    s["unique_invoices"]  = df["InvoiceNo"].nunique(dropna=True)  if "InvoiceNo"  in df.columns else pd.NA

    # Returns (credit notes 'C%')
    if invoice_no is not None:
        s["credit_invoices_cnt"] = invoice_no.str.startswith("C", na=False).sum()

    # Quantity diagnostics
    if "Quantity" in df.columns:
        s["qty_negative_cnt"] = (df["Quantity"] < 0).sum()
        s["qty_zero_cnt"]     = (df["Quantity"] == 0).sum()

    # UnitPrice diagnostics
    if "UnitPrice" in df.columns:
        s["unitprice_le0_cnt"] = (df["UnitPrice"] <= 0).sum()

    # Description missing/blank
    if description is not None:
        s["desc_missing_cnt"] = description.isna().sum()
        s["desc_blank_cnt"]   = description.fillna("").str.strip().eq("").sum()

    # CustomerID missing
    if customer_id is not None:
        miss = customer_id.isna().sum()
        s["customerid_missing_cnt"] = miss
        s["customerid_missing_pct"] = (miss / len(df) * 100) if len(df) else 0.0

    # IsReturn flag (if exists)
    if "IsReturn" in df.columns:
        s["isreturn_true_cnt"] = df["IsReturn"].sum()
        s["isreturn_true_pct"] = df["IsReturn"].mean() * 100

    # TotalPrice diagnostics (if exists)
    if "TotalPrice" in df.columns:
        s["totalprice_negative_cnt"] = (df["TotalPrice"] < 0).sum()
        s["totalprice_zero_cnt"]     = (df["TotalPrice"] == 0).sum()

    s.name = name
    return s


def compare_quality(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side comparison table (raw vs clean) with delta."""
    raw_s = data_quality_metrics(raw, "raw")
    clean_s = data_quality_metrics(clean, "clean")
    comp = pd.concat([raw_s, clean_s], axis=1)

    # Numeric delta (clean - raw) where possible
    def _delta(a: Any, b: Any) -> Any:
        try:
            return b - a
        except Exception:
            return pd.NA

    comp["delta"] = [_delta(comp.loc[idx, "raw"], comp.loc[idx, "clean"]) for idx in comp.index]
    return comp


if __name__ == "__main__":
    # ---- Load RAW
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing raw CSV at: {RAW_CSV_PATH.resolve()}")
    raw_df = pd.read_csv(RAW_CSV_PATH)

    # Parse types for checks (non-destructive)
    for col in ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country"]:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype("string")
    if "InvoiceDate" in raw_df.columns:
        raw_df["InvoiceDate"] = pd.to_datetime(raw_df["InvoiceDate"], errors="coerce")

    # ---- Load CLEAN
    if not CLEAN_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing clean CSV at: {CLEAN_CSV_PATH.resolve()}")
    clean_df = pd.read_csv(CLEAN_CSV_PATH, parse_dates=["InvoiceDate"])

    # ---- Detailed per-dataset reports
    check_parameters(raw_df, name="Raw Data")
    check_parameters(clean_df, name="Clean Data")

    # ---- Comparison table (raw vs clean + delta)
    print("\n\n===== Data Quality Comparison (raw vs clean) =====")
    comp = compare_quality(raw_df, clean_df)

    # Pretty print with formatted percentages for selected rows
    pct_rows = {"customerid_missing_pct", "isreturn_true_pct"}

    def _fmt(val: Any, idx: str) -> Any:
        # format only numeric values for known percentage rows
        if isinstance(val, (int, float)) and idx in pct_rows:
            return f"{val:.2f}%"
        return val

    pretty = comp.copy()
    for idx in pct_rows:
        if idx in pretty.index:
            pretty.loc[idx, "raw"]   = _fmt(pretty.loc[idx, "raw"], idx)
            pretty.loc[idx, "clean"] = _fmt(pretty.loc[idx, "clean"], idx)
            # leave 'delta' numeric (optionally format as % if desired)

    print(pretty)

    # (Optional) Save comparison to CSV for repo / Power BI
    out_dir = Path(r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\outputs")
    out_dir.mkdir(exist_ok=True)
    comp.to_csv(out_dir / "data_quality_comparison.csv", index=True, encoding="utf-8-sig")
    print(f"\nSaved comparison to {out_dir / 'data_quality_comparison.csv'}")
