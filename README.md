Podczas eksploracji zbioru Online Retail zidentyfikowałem kilka problemów jakościowych w danych:

1.InvoiceNo

Część faktur zaczyna się od litery C – oznaczają one faktury korekcyjne (zwroty).
Liczba takich faktur: X (≈ Y% wszystkich).

2.Quantity

Występują wartości ujemne, które odpowiadają zwróconym pozycjom.
Liczba rekordów z Quantity < 0: X.

3.UnitPrice

Znaleziono wartości równe 0 lub ujemne, co sugeruje błędy w danych lub wpisy testowe.
Liczba rekordów z UnitPrice <= 0: X.

4.CustomerID

Około X rekordów (~Y%) nie ma przypisanego identyfikatora klienta.
W analizie klientów takie rekordy należy traktować jako anonimowe lub pominąć.

5.Description

Część rekordów ma brakujące opisy produktów lub zawiera jedynie spacje.
Brakujące: X, puste: Y.
Można je uzupełnić placeholderem "(No description)".