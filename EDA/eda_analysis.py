import pandas as pd
from pathlib import Path
from typing import Any, Optional

class DataQuality:


    def __init__(self, raw_csv_path: Path, clean_csv_path: Path):
        self.RAW_CSV_PATH = Path(raw_csv_path)
        self.CLEAN_CSV_PATH = Path(clean_csv_path)

    # --- [1] helpers  ---

    @staticmethod
    def check_parameters(df: pd.DataFrame, name: str = "DataFrame") -> None:
        """Pretty console report with head / info / describe / missing + a few quick stats."""
        print(f"\n\n===== {name} =====")
        print("\nHead (5 rows):")
        print(df.head(5))

        print("\nInfo:")
        df.info()

        print("\nDescribe (numeric columns):")
        print(df.describe())

        print("\nMissing values per column:")
        print(df.isnull().sum())

        cols = df.columns
        if "CustomerID" in cols:
            print("\nUnique customers:", df["CustomerID"].nunique(dropna=True))
        if "StockCode" in cols:
            print("Unique products:", df["StockCode"].nunique(dropna=True))
        if "InvoiceNo" in cols:
            print("Unique invoices:", df["InvoiceNo"].nunique(dropna=True))
        if "InvoiceDate" in cols and pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
            print("Date range:", df["InvoiceDate"].min(), "→", df["InvoiceDate"].max())

    @staticmethod
    def _safe_string(series: pd.Series) -> pd.Series:
        """Ensure series is string dtype for string ops (like .str.startswith)."""
        if not pd.api.types.is_string_dtype(series):
            return series.astype("string")
        return series

    @classmethod
    def data_quality_metrics(cls, df: pd.DataFrame, name: str) -> pd.Series:
        """
        Compute key data-quality and business sanity metrics for Online Retail.
        Returns a pandas Series with named metrics.
        """
        s = pd.Series(dtype="object")  # allow mixed types (ints/floats/datetimes)

        s["rows"] = len(df)
        s["cols"] = df.shape[1]

        if "InvoiceDate" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
            df = df.copy()
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

        invoice_no = cls._safe_string(df["InvoiceNo"]) if "InvoiceNo" in df.columns else None
        description = cls._safe_string(df["Description"]) if "Description" in df.columns else None
        customer_id = df["CustomerID"] if "CustomerID" in df.columns else None

        if "InvoiceDate" in df.columns:
            s["date_min"] = pd.to_datetime(df["InvoiceDate"]).min()
            s["date_max"] = pd.to_datetime(df["InvoiceDate"]).max()

        s["unique_customers"] = df["CustomerID"].nunique(dropna=True) if "CustomerID" in df.columns else pd.NA
        s["unique_products"]  = df["StockCode"].nunique(dropna=True)  if "StockCode"  in df.columns else pd.NA
        s["unique_invoices"]  = df["InvoiceNo"].nunique(dropna=True)  if "InvoiceNo"  in df.columns else pd.NA

        if invoice_no is not None:
            s["credit_invoices_cnt"] = invoice_no.str.startswith("C", na=False).sum()

        if "Quantity" in df.columns:
            s["qty_negative_cnt"] = (df["Quantity"] < 0).sum()
            s["qty_zero_cnt"]     = (df["Quantity"] == 0).sum()

        if "UnitPrice" in df.columns:
            s["unitprice_le0_cnt"] = (df["UnitPrice"] <= 0).sum()

        if description is not None:
            s["desc_missing_cnt"] = description.isna().sum()
            s["desc_blank_cnt"]   = description.fillna("").str.strip().eq("").sum()

        if customer_id is not None:
            miss = customer_id.isna().sum()
            s["customerid_missing_cnt"] = miss
            s["customerid_missing_pct"] = (miss / len(df) * 100) if len(df) else 0.0

        if "IsReturn" in df.columns:
            s["isreturn_true_cnt"] = df["IsReturn"].sum()
            s["isreturn_true_pct"] = df["IsReturn"].mean() * 100

        if "TotalPrice" in df.columns:
            s["totalprice_negative_cnt"] = (df["TotalPrice"] < 0).sum()
            s["totalprice_zero_cnt"]     = (df["TotalPrice"] == 0).sum()

        s.name = name
        return s

    @classmethod
    def compare_quality(cls, raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
        """Build a side-by-side comparison table (raw vs clean) with delta."""
        raw_s = cls.data_quality_metrics(raw, "raw")
        clean_s = cls.data_quality_metrics(clean, "clean")
        comp = pd.concat([raw_s, clean_s], axis=1)

        def _delta(a: Any, b: Any) -> Any:
            try:
                return b - a
            except Exception:
                return pd.NA

        comp["delta"] = [_delta(comp.loc[idx, "raw"], comp.loc[idx, "clean"]) for idx in comp.index]
        return comp

    @classmethod
    def print_quality_for_readme(cls, df: pd.DataFrame, name: str = "dataset") -> None:
        """Print raw data-quality stats in README-friendly text form."""
        s = cls.data_quality_metrics(df, name)
        print(f"\n===== Data Quality Summary: {name} =====")
        if "credit_invoices_cnt" in s:
            print(f"Anulowane faktury (InvoiceNo zaczyna się na 'C'): {s['credit_invoices_cnt']}")
        if "qty_negative_cnt" in s:
            print(f"Rekordy z Quantity < 0: {s['qty_negative_cnt']}")
        if "qty_zero_cnt" in s:
            print(f"Rekordy z Quantity = 0: {s['qty_zero_cnt']}")
        if "unitprice_le0_cnt" in s:
            print(f"Rekordy z UnitPrice <= 0: {s['unitprice_le0_cnt']}")
        if "customerid_missing_cnt" in s:
            print(f"Rekordy bez CustomerID: {s['customerid_missing_cnt']} "
                  f"(~{s['customerid_missing_pct']:.2f}%)")
        if "desc_missing_cnt" in s and "desc_blank_cnt" in s:
            print(f"Brakujące opisy: {s['desc_missing_cnt']}, puste opisy: {s['desc_blank_cnt']}")



    def run(self) -> pd.DataFrame:
        """Odtwarza kolejność działań z Twojego __main__: ładowanie, printy i porównanie."""
        # Load RAW
        if not self.RAW_CSV_PATH.exists():
            raise FileNotFoundError(f"Missing raw CSV at: {self.RAW_CSV_PATH.resolve()}")
        raw_df = pd.read_csv(self.RAW_CSV_PATH)


        for col in ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country"]:
            if col in raw_df.columns:
                raw_df[col] = raw_df[col].astype("string")
        if "InvoiceDate" in raw_df.columns:
            raw_df["InvoiceDate"] = pd.to_datetime(raw_df["InvoiceDate"], errors="coerce")

        # README-friendly print
        self.print_quality_for_readme(raw_df, name="Raw Online Retail")

        # Load CLEAN
        if not self.CLEAN_CSV_PATH.exists():
            raise FileNotFoundError(f"Missing clean CSV at: {self.CLEAN_CSV_PATH.resolve()}")
        clean_df = pd.read_csv(self.CLEAN_CSV_PATH, parse_dates=["InvoiceDate"])

        # Detailed reports
        self.check_parameters(raw_df, name="Raw Data")
        self.check_parameters(clean_df, name="Clean Data")

        # Comparison table
        print("\n\n===== Data Quality Comparison (raw vs clean) =====")
        comp = self.compare_quality(raw_df, clean_df)

        # Pretty format
        pct_rows = {"customerid_missing_pct", "isreturn_true_pct"}

        def _fmt(val: Any, idx: str) -> Any:
            if isinstance(val, (int, float)) and idx in pct_rows:
                return f"{val:.2f}%"
            return val

        pretty = comp.copy()
        for idx in pct_rows:
            if idx in pretty.index:
                pretty.loc[idx, "raw"]   = _fmt(pretty.loc[idx, "raw"], idx)
                pretty.loc[idx, "clean"] = _fmt(pretty.loc[idx, "clean"], idx)

        print(pretty)
        return comp