import pandas as pd
import numpy as np

# Wczytaj czysty dataset
df = pd.read_csv(r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\EDA\online_retail_clean.csv", parse_dates=["InvoiceDate"])

# =====================
# 1) Agregacja na poziomie InvoiceNo (zamówienie)
# =====================
basket = df.groupby(["InvoiceNo", "CustomerID", "Country"]).agg(
    BasketSize=("Quantity", "sum"),
    UniqueProducts=("StockCode", "nunique"),
    AvgPrice=("UnitPrice", "mean"),
    MaxPrice=("UnitPrice", "max"),
    MinPrice=("UnitPrice", "min"),
    TotalValue=("TotalPrice", "sum"),
    CheapItemShare=("UnitPrice", lambda x: (x < 1).mean()),  # udział tanich produktów
    InvoiceDate=("InvoiceDate", "max"),  # data zamówienia
    IsReturn=("IsReturn", "max")  # jeśli jakakolwiek pozycja to zwrot → całe zamówienie
).reset_index()

# Diversity: liczba unikalnych produktów / całkowita liczba pozycji
basket["Diversity"] = basket["UniqueProducts"] / basket["BasketSize"]

# =====================
# 2) Feature engineering – cechy czasowe
# =====================
basket["Month"] = basket["InvoiceDate"].dt.month
basket["Weekday"] = basket["InvoiceDate"].dt.weekday
basket["Hour"] = basket["InvoiceDate"].dt.hour
basket["IsWeekend"] = basket["Weekday"].isin([5, 6]).astype(int)
basket["Quarter"] = basket["InvoiceDate"].dt.quarter

# =====================
# 3) Cechy klienta (historia wcześniejszych zamówień)
# =====================

# Posortuj po dacie, żeby liczyć historię
basket = basket.sort_values(by=["CustomerID", "InvoiceDate"])

# Funkcje pomocnicze: liczba wcześniejszych zamówień i zwrotów
basket["PastOrders"] = basket.groupby("CustomerID").cumcount()
basket["PastReturns"] = (
    basket.groupby("CustomerID")["IsReturn"].cumsum().shift(fill_value=0)
)
basket["ReturnRate"] = basket["PastReturns"] / basket["PastOrders"].replace(0, np.nan)

# Recency (dni od poprzedniego zakupu klienta)
basket["PrevDate"] = basket.groupby("CustomerID")["InvoiceDate"].shift()
basket["Recency"] = (basket["InvoiceDate"] - basket["PrevDate"]).dt.days
basket["Recency"] = basket["Recency"].fillna(basket["Recency"].median())

# =====================
# 4) Label (y) i Features (X)
# =====================
y = basket["IsReturn"].astype(int)

#5 wartosci absolutne/ zapobieganie data leakeage
basket["AbsBasketSize"] = basket["BasketSize"].abs()
basket["AbsTotalValue"] = basket["TotalValue"].abs()
basket["Diversity"] = basket["UniqueProducts"] / basket["AbsBasketSize"].replace(0, np.nan)


# Drop kolumn, które nie są feature’ami (identyfikatory i target)
X = basket.drop(columns=["InvoiceNo", "InvoiceDate", "IsReturn","CustomerID", "PrevDate","PastOrders","PastReturns","ReturnRate","BasketSize","TotalValue"])
X["Country"] = X["Country"].astype("category")
print("X shape:", X.shape)
print("y shape:", y.shape)
print(X.head())
print(y.value_counts(normalize=True))





#dates

# używamy dat z 'basket' do stworzenia masek, a tniemy X i y
cutoff = pd.Timestamp("2011-10-01")
train_idx = basket["InvoiceDate"] < cutoff
test_idx  = ~train_idx

X_train_chrono = X.loc[train_idx].copy()
X_test_chrono  = X.loc[test_idx].copy()
y_train_chrono = y.loc[train_idx].copy()
y_test_chrono  = y.loc[test_idx].copy()
print("Train size:", X_train_chrono.shape, "Test size:", X_test_chrono.shape)

X_train_chrono.info()

