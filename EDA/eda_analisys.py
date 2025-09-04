import pandas as pd

df = pd.read_csv("online_retail.csv")


def check_parameters(dataframe):
    print(dataframe.head(5))
    dataframe.info()
    dataframe.describe()
    dataframe.isnull().sum()

check_parameters(df)


# 1) InvoiceNo – faktury zaczynające się na "C" (zwroty)
zwroty = df["InvoiceNo"].str.startswith("C", na=False).sum()
print("Liczba faktur-zwrotów:", zwroty)

# 2) Quantity – wartości ujemne (pozycje zwrotne)
qty_neg = (df["Quantity"] < 0).sum()
print("Liczba pozycji ze zwrotem (Quantity < 0):", qty_neg)

# 3) UnitPrice – ceny 0 lub ujemne
price_zero_neg = (df["UnitPrice"] <= 0).sum()
print("Liczba pozycji z ceną <= 0:", price_zero_neg)

# 4) CustomerID – braki
cust_missing = df["CustomerID"].isna().sum()
print("Braki w CustomerID:", cust_missing, "/", len(df), "=", round(cust_missing/len(df)*100, 2), "%")

# 5) Description – brakujące lub tylko spacje
desc_missing = df["Description"].isna().sum()
desc_blank = df["Description"].fillna("").str.strip().eq("").sum()
print("Brakujące Description:", desc_missing)
print("Puste/tajne Description:", desc_blank)


df_clean = pd.read_csv("online_retail_clean.csv")
check_parameters(df_clean)