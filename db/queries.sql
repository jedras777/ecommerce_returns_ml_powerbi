-- Zwrotność per kategoria / produkt
SELECT p.description,
       SUM(CASE WHEN oi.quantity < 0 THEN 1 ELSE 0 END)::float / COUNT(*) AS return_rate
FROM order_items oi
JOIN products p ON oi.stock_code = p.stock_code
GROUP BY p.description
ORDER BY return_rate DESC
LIMIT 10;

-- Średnia wartość koszyka
SELECT o.invoice_no,
       SUM(oi.quantity * oi.unit_price) AS basket_value
FROM orders o
JOIN order_items oi ON o.invoice_no = oi.invoice_no
GROUP BY o.invoice_no
ORDER BY basket_value DESC
LIMIT 10;

-- Najbardziej wartościowi klienci
SELECT c.customer_id,
       SUM(oi.quantity * oi.unit_price) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.invoice_no = oi.invoice_no
GROUP BY c.customer_id
ORDER BY total_spent DESC
LIMIT 10;
