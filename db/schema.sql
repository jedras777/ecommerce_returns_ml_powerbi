CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    country VARCHAR(50)
);

CREATE TABLE products (
    stock_code VARCHAR(20) PRIMARY KEY,
    description TEXT,
    unit_price NUMERIC
);

CREATE TABLE orders (
    invoice_no VARCHAR(20) PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    invoice_date TIMESTAMP
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    invoice_no VARCHAR(20) REFERENCES orders(invoice_no),
    stock_code VARCHAR(20) REFERENCES products(stock_code),
    quantity INT,
    unit_price NUMERIC
);

CREATE TABLE returns (
    id SERIAL PRIMARY KEY,
    invoice_no VARCHAR(20),
    stock_code VARCHAR(20),
    quantity INT
);
