# DB Layer

Zawiera pliki SQL do pracy z danymi Online Retail.

## Pliki
- `schema.sql` – definicja tabel (`orders`, `order_items`, `products`, `returns`, `customers`).
- `queries.sql` – przykładowe zapytania biznesowe (zwrotność per kategoria, średnia wartość koszyka, top klienci wg RFM).

## Uruchomienie
1. Utwórz bazę w PostgreSQL:  
   `createdb retail`
2. Załaduj schemat:  
   `psql -d retail -f schema.sql`
3. Uruchom zapytania:  
   `psql -d retail -f queries.sql`
