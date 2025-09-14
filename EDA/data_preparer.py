import pandas as pd

EXCEL_PATH: str = r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\inputs\Online Retail.xlsx"
CSV_PATH: str = r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\outputs\online_retail.csv"
CLEAN_PATH: str = r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\outputs\online_retail_clean.csv"

class DataPreparer:

    def __init__(self, excel_path: str = EXCEL_PATH, csv_path: str = CSV_PATH, clean_path: str = CLEAN_PATH):
        self.excel_path = excel_path
        self.csv_path = csv_path
        self.clean_path = clean_path

    def excel_to_csv(self) -> str:
        """Load data from Excel (sheet 'Online Retail') and save as CSV."""
        df = pd.read_excel(self.excel_path, sheet_name="Online Retail")
        df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
        print(f"Excel converted to CSV: {self.csv_path}")
        return self.csv_path

    def clean_data(self) -> pd.DataFrame:
        """Perform data cleaning and export a cleaned CSV."""
        # Load raw CSV
        df = pd.read_csv(self.csv_path)

        # Cast selected columns to string type
        for col in ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country"]:
            df[col] = df[col].astype("string")

        # Convert InvoiceDate to datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        print("Date range:", df["InvoiceDate"].min(), "→", df["InvoiceDate"].max())

        # 1) Remove rows with UnitPrice <= 0
        df = df[df["UnitPrice"] > 0]

        # 2) Remove rows with Quantity == 0
        df = df[df["Quantity"] != 0]

        # 3) Mark returns: either negative Quantity or InvoiceNo starting with "C"
        df["IsReturn"] = df["InvoiceNo"].str.startswith("C", na=False) | (df["Quantity"] < 0)

        # 4) Handle CustomerID: fill missing with "Anonim"
        df["CustomerID"] = df["CustomerID"].fillna("Anonim").astype("string")

        # 5) Clean Description: fill NA, strip spaces, replace empty with placeholder
        df["Description"] = df["Description"].fillna("(No description)").astype("string")
        df["Description"] = df["Description"].str.strip()
        df.loc[df["Description"] == "", "Description"] = "(No description)"

        # 6) Re-parse InvoiceDate just in case
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

        # 7) Add TotalPrice (Quantity × UnitPrice)
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        # 8) Save cleaned data
        df.to_csv(self.clean_path, index=False, encoding="utf-8-sig")
        print(f"Cleaning finished. Saved to {self.clean_path}")
        df.info()

        return df
