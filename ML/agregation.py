import pandas as pd
import numpy as np


def load_clean_data(path: str) -> pd.DataFrame:
    """Load the cleaned dataset and parse InvoiceDate as datetime."""
    return pd.read_csv(path, parse_dates=["InvoiceDate"])


def build_order_level_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to the order (InvoiceNo) level with basket metrics,
    time-based features, and customer history features.
    """
    # --- 1) Basket-level aggregation ---
    basket = (
        df.groupby(["InvoiceNo", "CustomerID", "Country"])
        .agg(
            BasketSize=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
            AvgPrice=("UnitPrice", "mean"),
            MaxPrice=("UnitPrice", "max"),
            MinPrice=("UnitPrice", "min"),
            TotalValue=("TotalPrice", "sum"),
            CheapItemShare=("UnitPrice", lambda x: (x < 1).mean()),  # share of cheap items
            InvoiceDate=("InvoiceDate", "max"),  # order date
            IsReturn=("IsReturn", "max"),        # if any item is a return → whole order is marked
        )
        .reset_index()
    )

    # Diversity: number of unique products / total basket size
    basket["Diversity"] = basket["UniqueProducts"] / basket["BasketSize"]

    # --- 2) Time-based features ---
    basket["Month"] = basket["InvoiceDate"].dt.month
    basket["Weekday"] = basket["InvoiceDate"].dt.weekday
    basket["Hour"] = basket["InvoiceDate"].dt.hour
    basket["IsWeekend"] = basket["Weekday"].isin([5, 6]).astype(int)
    basket["Quarter"] = basket["InvoiceDate"].dt.quarter

    # --- 3) Customer history features ---
    # Sort by date per customer to calculate sequential history
    basket = basket.sort_values(by=["CustomerID", "InvoiceDate"])

    # Number of previous orders and returns
    basket["PastOrders"] = basket.groupby("CustomerID").cumcount()
    basket["PastReturns"] = (
        basket.groupby("CustomerID")["IsReturn"].cumsum().shift(fill_value=0)
    )
    basket["ReturnRate"] = basket["PastReturns"] / basket["PastOrders"].replace(0, np.nan)

    # Recency: days since previous order
    basket["PrevDate"] = basket.groupby("CustomerID")["InvoiceDate"].shift()
    basket["Recency"] = (basket["InvoiceDate"] - basket["PrevDate"]).dt.days
    basket["Recency"] = basket["Recency"].fillna(basket["Recency"].median())

    # --- 4) Leakage-safe absolute values ---
    basket["AbsBasketSize"] = basket["BasketSize"].abs()
    basket["AbsTotalValue"] = basket["TotalValue"].abs()
    basket["Diversity"] = basket["UniqueProducts"] / basket["AbsBasketSize"].replace(0, np.nan)

    return basket


def build_features_and_target(basket: pd.DataFrame):
    """Prepare feature matrix X and target vector y."""
    y = basket["IsReturn"].astype(int)

    # Drop non-feature columns
    X = basket.drop(
        columns=[
            "InvoiceNo", "InvoiceDate", "IsReturn",
            "CustomerID", "PrevDate", "PastOrders",
            "PastReturns", "ReturnRate", "BasketSize", "TotalValue",
        ]
    )
    X["Country"] = X["Country"].astype("category")
    return X, y


def chronological_split(X: pd.DataFrame, y: pd.Series, basket: pd.DataFrame, cutoff: str):
    """Split data into train/test sets chronologically by a cutoff date."""
    cutoff_date = pd.Timestamp(cutoff)
    train_idx = basket["InvoiceDate"] < cutoff_date
    test_idx = ~train_idx

    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()

    return X_train, X_test, y_train, y_test


# --- Prepare dataset on module load (so it's importable) ---

# Load data
df = load_clean_data(r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\outputs\online_retail_clean.csv")

# Aggregate orders
basket = build_order_level_aggregation(df)

# Build features and labels
X, y = build_features_and_target(basket)

# Chronological split
X_train, X_test, y_train, y_test = chronological_split(X, y, basket, cutoff="2011-10-01")

# Explicit export
__all__ = ["basket", "X", "y", "X_train", "X_test", "y_train", "y_test"]

# --- Debug only when run directly ---
if __name__ == "__main__":
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(X.head())
    print(y.value_counts(normalize=True))
    print("Train size:", X_train.shape, "Test size:", X_test.shape)
    X_train.info()
