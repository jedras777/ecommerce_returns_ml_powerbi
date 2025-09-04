import pandas as pd

# Wczytanie danych z Excela (zmień ścieżkę na swoją)
df = pd.read_excel("Online Retail.xlsx", sheet_name="Online Retail")

# Zapis do CSV
df.to_csv("online_retail.csv", index=False, encoding="utf-8-sig")
