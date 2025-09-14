# src/business_eda.py
import pandas as pd

class BusinessEDA:
    def __init__(self, df: pd.DataFrame):
        """Store a working copy of the dataset."""
        self.df = df.copy()

    def sales_over_time(self, verbose: bool = True):
        """Monthly and weekly sales (sum of TotalPrice). Returns (monthly, weekly)."""
        self.df["Month"] = self.df["InvoiceDate"].dt.to_period("M")
        monthly_sales = self.df.groupby("Month")["TotalPrice"].sum()

        self.df["Week"] = self.df["InvoiceDate"].dt.to_period("W")
        weekly_sales = self.df.groupby("Week")["TotalPrice"].sum()

        if verbose:
            print("Monthly sales:\n", monthly_sales)
            print("\nWeekly sales:\n", weekly_sales)

        return monthly_sales, weekly_sales

    def returns_analysis(self, verbose: bool = True):
        """Monthly return rate (%) and top countries by return rate. Returns (monthly_rate, top_countries)."""
        # assumes Month already created by sales_over_time; if not, create it
        if "Month" not in self.df.columns:
            self.df["Month"] = self.df["InvoiceDate"].dt.to_period("M")

        monthly_returns = self.df.groupby("Month")["IsReturn"].mean()  # 0..1
        returns_by_country = (
            self.df.groupby("Country")["IsReturn"].mean().sort_values(ascending=False)
        )

        if verbose:
            print("Monthly return rate (%):\n", (monthly_returns * 100).round(2))
            print("\nTop return rate by country:\n", returns_by_country.head(10))

        return monthly_returns, returns_by_country

    def products_analysis(self, top_n: int = 10, verbose: bool = True):
        """Top selling and top returned products. Returns (top_selling, top_returned)."""
        top_products = (
            self.df[self.df["IsReturn"] == False]
            .groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )

        top_returned = (
            self.df[self.df["IsReturn"] == True]
            .groupby("Description")["Quantity"]
            .sum()
            .sort_values()  # most negative (highest absolute returns) at top
            .head(top_n)
        )

        if verbose:
            print(f"Top {top_n} selling products:\n", top_products)
            print(f"\nTop {top_n} returned products:\n", top_returned)

        return top_products, top_returned

    def rfm_analysis(self, verbose: bool = True):
        """RFM table and Monetary-based segmentation. Returns (rfm_df, segment_counts)."""
        snapshot_date = self.df["InvoiceDate"].max() + pd.Timedelta(days=1)

        rfm = (
            self.df.groupby("CustomerID")
            .agg({
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
                "InvoiceNo": "nunique",                                   # Frequency
                "TotalPrice": "sum"                                       # Monetary
            })
            .rename(columns={
                "InvoiceDate": "Recency",
                "InvoiceNo": "Frequency",
                "TotalPrice": "Monetary"
            })
        )

        rfm["MonetarySegment"] = pd.qcut(rfm["Monetary"], q=3, labels=["Low", "Medium", "High"])
        seg_counts = rfm["MonetarySegment"].value_counts()

        if verbose:
            print("RFM sample:\n", rfm.head())
            print("\nCustomer segmentation by Monetary:\n", seg_counts)

        return rfm, seg_counts

    def run_all(self, verbose: bool = True):
        """Convenience: run the whole EDA block in sequence."""
        monthly, weekly = self.sales_over_time(verbose=verbose)
        mret, top_c = self.returns_analysis(verbose=verbose)
        top_s, top_r = self.products_analysis(verbose=verbose)
        rfm, seg = self.rfm_analysis(verbose=verbose)
        return {
            "monthly_sales": monthly,
            "weekly_sales": weekly,
            "monthly_return_rate": mret,
            "returns_by_country": top_c,
            "top_selling": top_s,
            "top_returned": top_r,
            "rfm": rfm,
            "segment_counts": seg,
        }
