import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, clean_path: str):
        """Path to the cleaned CSV with InvoiceDate parsable as datetime."""
        self.clean_path = clean_path
        self.df: pd.DataFrame | None = None
        self.basket: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None

    # 1) load
    def load_clean_data(self) -> pd.DataFrame:
        """Load the cleaned dataset and parse InvoiceDate as datetime."""
        self.df = pd.read_csv(self.clean_path, parse_dates=["InvoiceDate"])
        return self.df

    # 2) order-level aggregation
    def build_order_level_aggregation(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Aggregate to the order (InvoiceNo) level with basket metrics,
        time-based features, and customer history features.
        """
        if df is None:
            if self.df is None:
                raise ValueError("Call load_clean_data() first or pass df explicitly.")
            df = self.df

        basket = (
            df.groupby(["InvoiceNo", "CustomerID", "Country"])
            .agg(
                BasketSize=("Quantity", "sum"),
                UniqueProducts=("StockCode", "nunique"),
                AvgPrice=("UnitPrice", "mean"),
                MaxPrice=("UnitPrice", "max"),
                MinPrice=("UnitPrice", "min"),
                TotalValue=("TotalPrice", "sum"),
                CheapItemShare=("UnitPrice", lambda x: (x < 1).mean()),
                InvoiceDate=("InvoiceDate", "max"),
                IsReturn=("IsReturn", "max"),
            )
            .reset_index()
        )

        # Diversity based on raw and absolute basket size
        basket["Diversity"] = basket["UniqueProducts"] / basket["BasketSize"]

        # Time-based
        basket["Month"] = basket["InvoiceDate"].dt.month
        basket["Weekday"] = basket["InvoiceDate"].dt.weekday
        basket["Hour"] = basket["InvoiceDate"].dt.hour
        basket["IsWeekend"] = basket["Weekday"].isin([5, 6]).astype(int)
        basket["Quarter"] = basket["InvoiceDate"].dt.quarter

        # Sort to compute sequential history per customer
        basket = basket.sort_values(by=["CustomerID", "InvoiceDate"])

        basket["PastOrders"] = basket.groupby("CustomerID").cumcount()
        basket["PastReturns"] = basket.groupby("CustomerID")["IsReturn"].cumsum().shift(fill_value=0)
        basket["ReturnRate"] = basket["PastReturns"] / basket["PastOrders"].replace(0, np.nan)

        basket["PrevDate"] = basket.groupby("CustomerID")["InvoiceDate"].shift()
        basket["Recency"] = (basket["InvoiceDate"] - basket["PrevDate"]).dt.days
        basket["Recency"] = basket["Recency"].fillna(basket["Recency"].median())

        # Leakage-safe absolute values
        basket["AbsBasketSize"] = basket["BasketSize"].abs()
        basket["AbsTotalValue"] = basket["TotalValue"].abs()
        basket["Diversity"] = basket["UniqueProducts"] / basket["AbsBasketSize"].replace(0, np.nan)

        self.basket = basket
        return basket

    # 3) features & target
    def build_features_and_target(self, basket: pd.DataFrame | None = None):
        """Prepare feature matrix X and target vector y."""
        if basket is None:
            if self.basket is None:
                raise ValueError("Call build_order_level_aggregation() first or pass basket explicitly.")
            basket = self.basket

        y = basket["IsReturn"].astype(int)
        X = basket.drop(
            columns=[
                "InvoiceNo", "InvoiceDate", "IsReturn",
                "CustomerID", "PrevDate", "PastOrders",
                "PastReturns", "ReturnRate", "BasketSize", "TotalValue",
            ]
        )
        X["Country"] = X["Country"].astype("category")

        self.X, self.y = X, y
        return X, y

    # 4) chronological split
    @staticmethod
    def chronological_split(
        X: pd.DataFrame, y: pd.Series, basket: pd.DataFrame, cutoff: str
    ):
        """Split data into train/test sets chronologically by a cutoff date."""
        cutoff_date = pd.Timestamp(cutoff)
        train_idx = basket["InvoiceDate"] < cutoff_date
        test_idx = ~train_idx

        X_train = X.loc[train_idx].copy()
        X_test = X.loc[test_idx].copy()
        y_train = y.loc[train_idx].copy()
        y_test = y.loc[test_idx].copy()

        return X_train, X_test, y_train, y_test

    # 5) convenience runner
    def prepare(self, cutoff: str):
        """
        One-call convenience: load → aggregate → features → time split.
        Returns: basket, X, y, X_train, X_test, y_train, y_test
        """
        df = self.load_clean_data()
        basket = self.build_order_level_aggregation(df)
        X, y = self.build_features_and_target(basket)
        X_train, X_test, y_train, y_test = self.chronological_split(X, y, basket, cutoff=cutoff)
        return basket, X, y, X_train, X_test, y_train, y_test
