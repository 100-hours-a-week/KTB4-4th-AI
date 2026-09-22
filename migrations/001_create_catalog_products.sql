CREATE TABLE catalog_products (
    platform varchar(50) NOT NULL CHECK (length(btrim(platform)) > 0),
    external_id varchar(255) NOT NULL CHECK (length(btrim(external_id)) > 0),
    name varchar(300) NOT NULL CHECK (length(btrim(name)) > 0),
    price numeric(12, 2) CHECK (price >= 0),
    PRIMARY KEY (platform, external_id)
);
