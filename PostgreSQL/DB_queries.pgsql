CREATE TABLE supplies (
  id INT PRIMARY KEY,
  name VARCHAR,
  description VARCHAR,
  manufacturer VARCHAR,
  color VARCHAR,
  inventory int CHECK (inventory > 0)
);